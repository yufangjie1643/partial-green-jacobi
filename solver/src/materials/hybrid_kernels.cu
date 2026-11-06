#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
#include "fi_modal.cuh"

__device__ __forceinline__ int wrap(int x,int n) { return x<0?x+n:(x>=n?x-n:x); }
template<bool indirect>
__device__ __forceinline__ const double* material(const double* c,const int32_t* map,int64_t e) {
  if constexpr(indirect) return c+int64_t(map[e])*21;
  else return c+e*21;
}
__device__ __forceinline__ void mul(const double* c,const double* e,double* s) {
  s[0]=c[0]*e[0]+c[1]*e[1]+c[2]*e[2]+c[3]*e[3]+c[4]*e[4]+c[5]*e[5];
  s[1]=c[1]*e[0]+c[6]*e[1]+c[7]*e[2]+c[8]*e[3]+c[9]*e[4]+c[10]*e[5];
  s[2]=c[2]*e[0]+c[7]*e[1]+c[11]*e[2]+c[12]*e[3]+c[13]*e[4]+c[14]*e[5];
  s[3]=c[3]*e[0]+c[8]*e[1]+c[12]*e[2]+c[15]*e[3]+c[16]*e[4]+c[17]*e[5];
  s[4]=c[4]*e[0]+c[9]*e[1]+c[13]*e[2]+c[16]*e[3]+c[18]*e[4]+c[19]*e[5];
  s[5]=c[5]*e[0]+c[10]*e[1]+c[14]*e[2]+c[17]*e[3]+c[19]*e[4]+c[20]*e[5];
}
template<bool indirect>
__global__ void apply_kernel(const double* u,const double* c,const int32_t* map,double* out,
    int nx,int ny,int nz,double gx,double gy,double gz,double volume) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(e>=int64_t(nx)*ny*nz)return;
  int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
  double local[24],f[24];
  const double g[4]={gx,gy,gz,volume};
  #pragma unroll
  for(int a=0;a<8;++a) {
    int xx=wrap(x+(a&1),nx),yy=wrap(y+((a>>1)&1),ny),zz=wrap(z+((a>>2)&1),nz);
    int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d)local[3*a+d]=u[3*node+d];
  }
  fi_modal(local,material<indirect>(c,map,e),0.,g,f);
  #pragma unroll
  for(int a=0;a<8;++a) {
    int xx=wrap(x+(a&1),nx),yy=wrap(y+((a>>1)&1),ny),zz=wrap(z+((a>>2)&1),nz);
    int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d)atomicAdd(out+3*node+d,f[3*a+d]);
  }
}
template<bool indirect>
__global__ void rhs_kernel(const double* c,const int32_t* map,const double* macro,double* out,
    int nx,int ny,int nz,double gx,double gy,double gz,double volume) {
  int64_t node=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(node>=int64_t(nx)*ny*nz)return;
  int x=node%nx,y=(node/nx)%ny,z=node/(int64_t(nx)*ny);
  double f[3]={0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;++a) {
    int ex=wrap(x-(a&1),nx),ey=wrap(y-((a>>1)&1),ny),ez=wrap(z-((a>>2)&1),nz);
    int64_t e=(int64_t(ez)*ny+ey)*nx+ex;
    double s[6];mul(material<indirect>(c,map,e),macro,s);
    double dx=((a&1)?gx:-gx),dy=((a&2)?gy:-gy),dz=((a&4)?gz:-gz);
    f[0]-=volume*(dx*s[0]+dy*s[5]+dz*s[4]);
    f[1]-=volume*(dx*s[5]+dy*s[1]+dz*s[3]);
    f[2]-=volume*(dx*s[4]+dy*s[3]+dz*s[2]);
  }
  #pragma unroll
  for(int d=0;d<3;++d)out[3*node+d]=f[d];
}
template<bool indirect>
__global__ void stress_kernel(const double* u,const double* c,const int32_t* map,
    const double* macro,double* partial,int nx,int ny,int nz,double gx,double gy,double gz) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int64_t n=int64_t(nx)*ny*nz;
  double s[6]={0.,0.,0.,0.,0.,0.};
  if(e<n) {
    int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
    double eps[6];
    #pragma unroll
    for(int j=0;j<6;++j)eps[j]=macro[j];
    #pragma unroll
    for(int a=0;a<8;++a) {
      int xx=wrap(x+(a&1),nx),yy=wrap(y+((a>>1)&1),ny),zz=wrap(z+((a>>2)&1),nz);
      int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
      double ux=u[3*node],uy=u[3*node+1],uz=u[3*node+2];
      double dx=((a&1)?gx:-gx),dy=((a&2)?gy:-gy),dz=((a&4)?gz:-gz);
      eps[0]+=dx*ux;eps[1]+=dy*uy;eps[2]+=dz*uz;
      eps[3]+=dz*uy+dy*uz;eps[4]+=dz*ux+dx*uz;eps[5]+=dy*ux+dx*uy;
    }
    mul(material<indirect>(c,map,e),eps,s);
  }
  __shared__ double shared[6][128];
  #pragma unroll
  for(int j=0;j<6;++j)shared[j][threadIdx.x]=s[j];
  __syncthreads();
  for(int stride=64;stride;stride>>=1) {
    if(threadIdx.x<stride) {
      #pragma unroll
      for(int j=0;j<6;++j)shared[j][threadIdx.x]+=shared[j][threadIdx.x+stride];
    }
    __syncthreads();
  }
  if(threadIdx.x==0) {
    #pragma unroll
    for(int j=0;j<6;++j)partial[int64_t(j)*gridDim.x+blockIdx.x]=shared[j][0];
  }
}
void check(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat64&&t.is_contiguous(),"CUDA contiguous FP64 required");
}
void check_material(const torch::Tensor& c,const torch::Tensor& map,bool indirect,int64_t n) {
  check(c);TORCH_CHECK(c.numel()%21==0,"C21 pool required");
  TORCH_CHECK(map.is_cuda()&&map.is_contiguous()&&map.scalar_type()==torch::kInt32&&map.device()==c.device(),"int32 CUDA row map required");
  TORCH_CHECK(indirect?map.numel()==n:c.numel()==21*n,"material shape mismatch");
}
} // namespace

torch::Tensor apply(torch::Tensor c,torch::Tensor map,torch::Tensor u,int nz,int ny,int nx,
    double hx,double hy,double hz,bool indirect) {
  int64_t n=int64_t(nx)*ny*nz;check_material(c,map,indirect,n);check(u);
  TORCH_CHECK(nx>=2&&ny>=2&&nz>=2&&u.numel()==3*n&&u.device()==c.device(),"grid/displacement mismatch");
  TORCH_CHECK(hx>0&&hy>0&&hz>0,"positive voxel sizes required");
  c10::cuda::CUDAGuard guard(u.device());auto out=torch::zeros_like(u);
  auto stream=at::cuda::getCurrentCUDAStream();int blocks=(n+127)/128;
  if(indirect)apply_kernel<true><<<blocks,128,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),map.data_ptr<int32_t>(),out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
  else apply_kernel<false><<<blocks,128,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),nullptr,out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor rhs(torch::Tensor c,torch::Tensor map,torch::Tensor macro,int nz,int ny,int nx,
    double hx,double hy,double hz,bool indirect) {
  int64_t n=int64_t(nx)*ny*nz;check_material(c,map,indirect,n);check(macro);
  TORCH_CHECK(macro.numel()==6&&macro.device()==c.device(),"six macro strain components required");
  c10::cuda::CUDAGuard guard(c.device());auto out=torch::empty({1,3*n},c.options());
  auto stream=at::cuda::getCurrentCUDAStream();int blocks=(n+127)/128;
  if(indirect)rhs_kernel<true><<<blocks,128,0,stream>>>(c.data_ptr<double>(),map.data_ptr<int32_t>(),macro.data_ptr<double>(),out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
  else rhs_kernel<false><<<blocks,128,0,stream>>>(c.data_ptr<double>(),nullptr,macro.data_ptr<double>(),out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor stress(torch::Tensor c,torch::Tensor map,torch::Tensor u,torch::Tensor macro,int nz,int ny,int nx,
    double hx,double hy,double hz,bool indirect) {
  int64_t n=int64_t(nx)*ny*nz;check_material(c,map,indirect,n);check(macro);check(u);
  TORCH_CHECK(u.numel()==3*n&&macro.numel()==6&&u.device()==c.device()&&macro.device()==c.device(),"field/device mismatch");
  c10::cuda::CUDAGuard guard(c.device());int blocks=(n+127)/128;
  auto partial=torch::empty({6,blocks},c.options());auto stream=at::cuda::getCurrentCUDAStream();
  if(indirect)stress_kernel<true><<<blocks,128,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),map.data_ptr<int32_t>(),macro.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz));
  else stress_kernel<false><<<blocks,128,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),nullptr,macro.data_ptr<double>(),partial.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return partial.sum(1).reshape({1,6})/double(n);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("rhs",&rhs);m.def("stress",&stress);}
