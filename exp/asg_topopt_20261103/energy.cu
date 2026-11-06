#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace {
#include "fi_modal.cuh"
__global__ void local_energy(const double* u,const double* c,const double* affine,double* out,int n) {
  int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x, count=int64_t(n)*n*n;
  if(e>=count)return;
  int load=blockIdx.y,x=e%n,y=e/n%n,z=e/(n*n);
  double v[24],f[24],h=1./n,g[4]={n/4.,n/4.,n/4.,h*h*h};
  #pragma unroll
  for(int a=0;a<8;++a) {
    int xx=(x+(a&1))%n, yy=(y+((a>>1)&1))%n, zz=(z+((a>>2)&1))%n;
    int64_t node=(int64_t(zz)*n+yy)*n+xx;
    #pragma unroll
    for(int d=0;d<3;++d)v[3*a+d]=u[int64_t(load)*3*count+3*node+d]+affine[24*load+3*a+d];
  }
  fi_modal(v,c,0.,g,f);
  double energy=0.;
  #pragma unroll
  for(int k=0;k<24;++k)energy+=v[k]*f[k];
  out[int64_t(load)*count+e]=energy/(h*h*h);
}
}
torch::Tensor evaluate(torch::Tensor u,torch::Tensor c,torch::Tensor affine,int n) {
  const int64_t count=int64_t(n)*n*n;
  TORCH_CHECK(n>=2&&u.is_cuda()&&u.is_contiguous()&&u.scalar_type()==torch::kFloat64&&u.numel()==9*count,"invalid displacements");
  TORCH_CHECK(c.is_cuda()&&c.is_contiguous()&&c.scalar_type()==torch::kFloat64&&c.numel()==21&&c.device()==u.device(),"invalid C0");
  TORCH_CHECK(affine.is_cuda()&&affine.is_contiguous()&&affine.scalar_type()==torch::kFloat64&&affine.numel()==72&&affine.device()==u.device(),"invalid affine field");
  c10::cuda::CUDAGuard guard(u.device());
  auto out=torch::empty({3,count},u.options());
  local_energy<<<dim3((count+127)/128,3),128,0,at::cuda::getCurrentCUDAStream()>>>(u.data_ptr<double>(),c.data_ptr<double>(),affine.data_ptr<double>(),out.data_ptr<double>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("evaluate",&evaluate);}
