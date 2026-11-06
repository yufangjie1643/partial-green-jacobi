#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
#include "modal_functions.cuh"

__device__ __forceinline__ int wrap(int v, int n) { return v >= n ? v-n : v; }

template<bool fi, bool energy>
__global__ void apply_kernel(const double* u,const double* c,const double* scale,
    double* output,double* partial,int nx,int ny,int nz,double hx,double hy,double hz) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t n=int64_t(nx)*ny*nz;
  double value=0.;
  if(e<n) {
    const int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
    double local[24],f[24];
    int64_t nodes[8];
    #pragma unroll
    for(int a=0;a<8;++a) {
      nodes[a]=(int64_t(wrap(z+((a>>2)&1),nz))*ny+wrap(y+((a>>1)&1),ny))*nx+wrap(x+(a&1),nx);
      #pragma unroll
      for(int d=0;d<3;++d)local[3*a+d]=u[3*nodes[a]+d];
    }
    const double g[4]={1./(4*hx),1./(4*hy),1./(4*hz),hx*hy*hz};
    if constexpr(fi)value=fi_eval<energy>(local,c+21*e,0.,g,f);
    else value=hg_eval<energy>(local,c+21*e,scale[e],g,f);
    #pragma unroll
    for(int a=0;a<8;++a) {
      #pragma unroll
      for(int d=0;d<3;++d)atomicAdd(output+3*nodes[a]+d,f[3*a+d]);
    }
  }
  if constexpr(energy) {
    for(int offset=16;offset>0;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    __shared__ double sums[8];
    if((threadIdx.x&31)==0)sums[threadIdx.x/32]=value;
    __syncthreads();
    if(threadIdx.x==0) {
      double total=0.;for(int i=0;i<blockDim.x/32;++i)total+=sums[i];
      partial[blockIdx.x]=total;
    }
  }
}

__global__ void scale_kernel(const double* c,double* s,int64_t n,double factor) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=n)return;
  c+=21*e;
  const double normal=c[0]+c[6]+c[11];
  const double first=normal+2*(c[1]+c[2]+c[7]);
  const double second=normal+2*(c[15]+c[18]+c[20]);
  s[e]=fmax((first+7*second)/60.,1e-12)*factor;
}

__global__ void coefficient_kernel(const double* tx,const double* ty,const double* tz,
    float* coef,int nx,int ny,int nz,double hx,double hy,double hz,double lam,double mu,double penalty) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int ns=nx/2+1;
  const int64_t count=int64_t(ns)*ny*nz;
  if(k>=count)return;
  const int x=k%ns,y=(k/ns)%ny,z=k/(int64_t(ns)*ny);
  const double sx=tx[2*x],cx=tx[2*x+1],sy=ty[2*y],cy=ty[2*y+1],sz=tz[2*z],cz=tz[2*z+1];
  const double dx=2/hx*sx*cy*cz,dy=2/hy*cx*sy*cz,dz=2/hz*cx*cy*sz;
  const double norm=dx*dx+dy*dy+dz*dz,volume=hx*hy*hz;
  const double shift=8*penalty*(sx*sx*sy*sy*cz*cz+sx*sx*cy*cy*sz*sz+cx*cx*sy*sy*sz*sz+sx*sx*sy*sy*sz*sz);
  const double a=volume*mu*norm+shift,b=volume*(lam+mu);
  coef[k]=k==0?0.:float(1/a);
  coef[count+k]=k==0?0.:float(b/(a+b*norm));
}

__global__ void spectral_kernel(const float* coef,const double* tx,const double* ty,const double* tz,
    const c10::complex<float>* r,c10::complex<float>* out,int nx,int ny,int nz,double hx,double hy,double hz) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int ns=nx/2+1;
  const int64_t count=int64_t(ns)*ny*nz;
  if(k>=count)return;
  const int x=k%ns,y=(k/ns)%ny,z=k/(int64_t(ns)*ny);
  const float dx=float(2/hx*tx[2*x]*ty[2*y+1]*tz[2*z+1]);
  const float dy=float(2/hy*tx[2*x+1]*ty[2*y]*tz[2*z+1]);
  const float dz=float(2/hz*tx[2*x+1]*ty[2*y+1]*tz[2*z]);
  const auto dot=dx*r[k]+dy*r[count+k]+dz*r[2*count+k];
  const float a=coef[k],b=coef[count+k];
  out[k]=a*(r[k]-(b*dx)*dot);
  out[count+k]=a*(r[count+k]-(b*dy)*dot);
  out[2*count+k]=a*(r[2*count+k]-(b*dz)*dot);
}
}

std::vector<torch::Tensor> apply_energy(torch::Tensor c,torch::Tensor scales,torch::Tensor u,
    int nz,int ny,int nx,double hx,double hy,double hz,bool fi,bool energy,int block) {
  TORCH_CHECK(c.is_cuda()&&u.is_cuda()&&c.scalar_type()==torch::kFloat64&&u.scalar_type()==torch::kFloat64,"CUDA FP64 required");
  TORCH_CHECK(c.is_contiguous()&&u.is_contiguous()&&c.numel()==int64_t(nx)*ny*nz*21,"B=1 contiguous C21 required");
  TORCH_CHECK(block==128||block==256,"block must be 128 or 256");
  c10::cuda::CUDAGuard guard(u.device());
  const int blocks=(int64_t(nx)*ny*nz+block-1)/block;
  auto out=torch::zeros_like(u),partial=torch::empty({energy?blocks:0},u.options());
  const auto stream=at::cuda::getCurrentCUDAStream();
  if(fi) {
    if(energy)apply_kernel<true,true><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),out.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,hx,hy,hz);
    else apply_kernel<true,false><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),out.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,hx,hy,hz);
  } else {
    if(energy)apply_kernel<false,true><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),out.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,hx,hy,hz);
    else apply_kernel<false,false><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),out.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,hx,hy,hz);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,partial};
}

torch::Tensor scales(torch::Tensor c,double factor) {
  const int64_t n=c.numel()/21;
  auto s=torch::empty({1,n},c.options());
  scale_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(c.data_ptr<double>(),s.data_ptr<double>(),n,factor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return s;
}

torch::Tensor coefficients(torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,
    int nx,int ny,int nz,double hx,double hy,double hz,double lam,double mu,double penalty) {
  const int64_t count=int64_t(nx/2+1)*ny*nz;
  auto out=torch::empty({2,count},tx.options().dtype(torch::kFloat32));
  coefficient_kernel<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),out.data_ptr<float>(),nx,ny,nz,hx,hy,hz,lam,mu,penalty);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor spectral(torch::Tensor coef,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,torch::Tensor r,
    int nx,int ny,int nz,double hx,double hy,double hz) {
  auto out=torch::empty_like(r);
  const int64_t count=int64_t(nx/2+1)*ny*nz;
  TORCH_CHECK(r.is_contiguous()&&r.scalar_type()==torch::kComplexFloat&&r.numel()==3*count,"B=1 complex64 spectrum required");
  spectral_kernel<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(coef.data_ptr<float>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),r.data_ptr<c10::complex<float>>(),out.data_ptr<c10::complex<float>>(),nx,ny,nz,hx,hy,hz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("apply_energy",&apply_energy);m.def("scales",&scales);
  m.def("coefficients",&coefficients);m.def("spectral",&spectral);
}
