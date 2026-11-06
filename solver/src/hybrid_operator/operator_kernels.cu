#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
#include "fi_modal.cuh"
#include "fi_isotropic.cuh"
__device__ __forceinline__ int wrap(int x,int n){return x<0?x+n:(x>=n?x-n:x);}
#include "background_grouped.cuh"

template<bool contrast,bool cached=false>
__global__ void element_kernel(const double* u,const double* pool,const int32_t* map,
    const uint8_t* flags,const int32_t* active,int64_t count,int shared,int base,double* out,
    int nx,int ny,int nz,double gx,double gy,double gz,double volume) {
  int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=count)return;
  int64_t e=contrast?int64_t(active[k]):k;
  int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
  int row=map[e];
  const double* c=pool+int64_t(row)*21;
  double local[24],f[24];const double g[4]={gx,gy,gz,volume};
  #pragma unroll
  for(int a=0;a<8;++a){
    int xx=wrap(x+(a&1),nx),yy=wrap(y+((a>>1)&1),ny),zz=wrap(z+((a>>2)&1),nz);
    int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d)local[3*a+d]=u[3*node+d];
  }
  if constexpr(contrast && !cached){
    double delta[21];
    #pragma unroll
    for(int j=0;j<21;++j)delta[j]=c[j]-pool[int64_t(base)*21+j];
    fi_modal(local,delta,0.,g,f);
  }else if constexpr(cached){
    fi_modal(local,c,0.,g,f);
  }else{
    if(row<shared && flags[row])fi_isotropic(local,c,0.,g,f);
    else fi_modal(local,c,0.,g,f);
  }
  #pragma unroll
  for(int a=0;a<8;++a){
    int xx=wrap(x+(a&1),nx),yy=wrap(y+((a>>1)&1),ny),zz=wrap(z+((a>>2)&1),nz);
    int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d)atomicAdd(out+3*node+d,f[3*a+d]);
  }
}

template<bool isotropic>
__global__ void background_kernel(const double* u,const double* S,double* out,int nx,int ny,int nz){
  int64_t node=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(node>=int64_t(nx)*ny*nz)return;
  int x=node%nx,y=(node/nx)%ny,z=node/(int64_t(nx)*ny);
  // Same lexicographic offset order as background.OFFSETS (x,y,z).
  constexpr int dx[13]={-1,-1,-1,-1,0,0,0,0,1,1,1,1,1};
  constexpr int dy[13]={-1,0,1,1,-1,0,1,1,-1,0,0,1,1};
  constexpr int dz[13]={1,1,0,1,1,1,0,1,1,0,1,0,1};
  double center[3]={u[3*node],u[3*node+1],u[3*node+2]},f[3]={0.,0.,0.};
  #pragma unroll
  for(int j=0;j<13;++j){
    int xp=wrap(x+dx[j],nx),yp=wrap(y+dy[j],ny),zp=wrap(z+dz[j],nz);
    int xm=wrap(x-dx[j],nx),ym=wrap(y-dy[j],ny),zm=wrap(z-dz[j],nz);
    int64_t np=(int64_t(zp)*ny+yp)*nx+xp,nm=(int64_t(zm)*ny+ym)*nx+xm;
    double v[3];
    #pragma unroll
    for(int d=0;d<3;++d)v[d]=(u[3*np+d]-center[d])+(u[3*nm+d]-center[d]);
    #pragma unroll
    for(int a=0;a<3;++a){
      #pragma unroll
      for(int b=0;b<3;++b){
        if constexpr(isotropic){
          int da=a==0?dx[j]:(a==1?dy[j]:dz[j]);
          int db=b==0?dx[j]:(b==1?dy[j]:dz[j]);
          if(a==b || (da!=0 && db!=0))f[a]+=S[j*9+3*a+b]*v[b];
        }else f[a]+=S[j*9+3*a+b]*v[b];
      }
    }
  }
  #pragma unroll
  for(int d=0;d<3;++d)out[3*node+d]=f[d];
}
void fp64(const torch::Tensor& t){TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat64&&t.is_contiguous(),"CUDA FP64 contiguous required");}
} // namespace

torch::Tensor apply(torch::Tensor pool,torch::Tensor map,torch::Tensor flags,torch::Tensor active,
    torch::Tensor stencil,torch::Tensor u,int nz,int ny,int nx,double hx,double hy,double hz,
    int base,int mode,bool isotropic,int block){
  fp64(pool);fp64(u);fp64(stencil);int64_t n=int64_t(nx)*ny*nz;
  TORCH_CHECK(nx>=2&&ny>=2&&nz>=2&&hx>0&&hy>0&&hz>0,"valid geometry required");
  TORCH_CHECK(u.numel()==3*n&&pool.numel()%21==0,"shape mismatch");
  for(const auto& t:{map,active})TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kInt32&&t.is_contiguous()&&t.device()==u.device(),"int32 CUDA index required");
  TORCH_CHECK(map.numel()==n&&flags.is_cuda()&&flags.scalar_type()==torch::kUInt8&&flags.is_contiguous(),"map/flags shape");
  TORCH_CHECK(pool.device()==u.device()&&flags.device()==u.device()&&stencil.device()==u.device(),"device mismatch");
  TORCH_CHECK(block==64||block==128||block==256,"block size");
  TORCH_CHECK(mode>=0&&mode<=3,"operator mode");
  c10::cuda::CUDAGuard guard(u.device());auto stream=at::cuda::getCurrentCUDAStream();
  auto out=torch::empty_like(u);
  if(mode==0){
    out.zero_();
    element_kernel<false><<<(n+block-1)/block,block,0,stream>>>(u.data_ptr<double>(),pool.data_ptr<double>(),map.data_ptr<int32_t>(),flags.data_ptr<uint8_t>(),nullptr,n,flags.numel(),0,out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
  }else{
    TORCH_CHECK(base>=0&&base<flags.numel()&&stencil.numel()==13*9,"background phase/stencil");
    if(isotropic && mode==3)background_grouped<<<(n+127)/128,128,0,stream>>>(u.data_ptr<double>(),stencil.data_ptr<double>(),out.data_ptr<double>(),nx,ny,nz);
    else if(isotropic)background_kernel<true><<<(n+127)/128,128,0,stream>>>(u.data_ptr<double>(),stencil.data_ptr<double>(),out.data_ptr<double>(),nx,ny,nz);
    else background_kernel<false><<<(n+127)/128,128,0,stream>>>(u.data_ptr<double>(),stencil.data_ptr<double>(),out.data_ptr<double>(),nx,ny,nz);
    int64_t count=active.numel();
    if(count){
      if(mode>=2)element_kernel<true,true><<<(count+block-1)/block,block,0,stream>>>(u.data_ptr<double>(),pool.data_ptr<double>(),map.data_ptr<int32_t>(),flags.data_ptr<uint8_t>(),active.data_ptr<int32_t>(),count,flags.numel(),base,out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
      else element_kernel<true><<<(count+block-1)/block,block,0,stream>>>(u.data_ptr<double>(),pool.data_ptr<double>(),map.data_ptr<int32_t>(),flags.data_ptr<uint8_t>(),active.data_ptr<int32_t>(),count,flags.numel(),base,out.data_ptr<double>(),nx,ny,nz,1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
pybind11::dict attributes(){
  pybind11::dict result;
  for(int k=0;k<6;++k){cudaFuncAttributes a;
    if(k==0)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<false>));
    else if(k==1)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true>));
    else if(k==2)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,background_kernel<true>));
    else if(k==3)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,background_kernel<false>));
    else if(k==4)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true,true>));
    else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,background_grouped));
    pybind11::dict d;d["registers"]=a.numRegs;d["local_bytes"]=a.localSizeBytes;d["shared_bytes"]=a.sharedSizeBytes;
    result[pybind11::str(k==0?"isotropic":k==1?"contrast":k==2?"background_isotropic":k==3?"background_general":k==4?"contrast_cached":"background_grouped")]=d;
  }return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("attributes",&attributes);}
