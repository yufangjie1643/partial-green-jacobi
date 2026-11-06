// Application specialization: C(x)=a(x)*C0, lambda=4/3, mu=1.
// The exact archived FI modal element formula is retained. Factoring out a
// lets the compiler eliminate the zero anisotropic coefficients and allows
// three loads in one launch. It does not construct a global stiffness matrix.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>

namespace {
#include "fi_modal.cuh"
__device__ __forceinline__ int wrap(int x,int n){return x>=n?x-n:x;}
__global__ void apply_scalar(const double* a,const double* u,const bool* active,double* out,int n){
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int load=blockIdx.y;int64_t count=int64_t(n)*n*n;
  if(e>=count||!active[load])return;
  u+=int64_t(load)*3*count;out+=int64_t(load)*3*count;
  int x=e%n,y=(e/n)%n,z=e/(n*n);
  const double c[21]={4./3.+2.,4./3.,4./3.,0.,0.,0.,4./3.+2.,4./3.,0.,0.,0.,4./3.+2.,0.,0.,0.,1.,0.,0.,1.,0.,1.};
  double h=1./n;const double g[4]={1./(4*h),1./(4*h),1./(4*h),h*h*h};
  double local[24],f[24];int64_t nodes[8];
  #pragma unroll
  for(int v=0;v<8;++v){
    nodes[v]=(int64_t(wrap(z+((v>>2)&1),n))*n+wrap(y+((v>>1)&1),n))*n+wrap(x+(v&1),n);
    #pragma unroll
    for(int d=0;d<3;++d)local[3*v+d]=u[3*nodes[v]+d];
  }
  fi_modal(local,c,0.,g,f);
  double coefficient=a[e];
  #pragma unroll
  for(int v=0;v<8;++v){
    #pragma unroll
    for(int d=0;d<3;++d)atomicAdd(out+3*nodes[v]+d,coefficient*f[3*v+d]);
  }
}
void check(const torch::Tensor& x){TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kFloat64,"contiguous CUDA FP64 required");}
__global__ void spectral_three(c10::complex<double>* spectrum,const double* c,int64_t count){
  int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=count)return;
  auto* r=spectrum+int64_t(blockIdx.y)*3*count;
  auto x=r[k],y=r[count+k],z=r[2*count+k];
  double c0=c[k],c1=c[count+k],c2=c[2*count+k],c3=c[3*count+k],c4=c[4*count+k],c5=c[5*count+k];
  r[k]=c0*x+c3*y+c4*z;r[count+k]=c3*x+c1*y+c5*z;r[2*count+k]=c4*x+c5*y+c2*z;
}
}
torch::Tensor apply(torch::Tensor a,torch::Tensor u,torch::Tensor active,int n){
  check(a);check(u);int64_t count=int64_t(n)*n*n;
  TORCH_CHECK(n>=2&&a.numel()==count&&u.dim()==2&&(u.size(0)==1||u.size(0)==3)&&u.size(1)==3*count&&a.device()==u.device(),"scalar FI shape/device mismatch");
  TORCH_CHECK(active.is_cuda()&&active.is_contiguous()&&active.device()==u.device()&&active.scalar_type()==torch::kBool&&active.numel()==u.size(0),"one CUDA bool mask per load required");
  c10::cuda::CUDAGuard guard(u.device());auto out=torch::zeros_like(u);
  apply_scalar<<<dim3((count+127)/128,u.size(0)),128,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<double>(),u.data_ptr<double>(),active.data_ptr<bool>(),out.data_ptr<double>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void spectral(torch::Tensor spectrum,torch::Tensor coefficient){
  check(coefficient);
  TORCH_CHECK(spectrum.is_cuda()&&spectrum.is_contiguous()&&spectrum.scalar_type()==torch::kComplexDouble&&spectrum.device()==coefficient.device(),"contiguous CUDA complex128 spectrum required");
  int64_t count=coefficient.numel()/6;
  TORCH_CHECK(count>0&&coefficient.numel()==6*count&&spectrum.numel()==9*count,"three-load spectral shape mismatch");
  c10::cuda::CUDAGuard guard(spectrum.device());
  spectral_three<<<dim3((count+255)/256,3),256,0,at::cuda::getCurrentCUDAStream()>>>(spectrum.data_ptr<c10::complex<double>>(),coefficient.data_ptr<double>(),count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("spectral",&spectral);}
