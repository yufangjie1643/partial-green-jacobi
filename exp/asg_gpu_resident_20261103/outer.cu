#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void gaussian_axis(const double* x, const double* weights, double* y,
                             int n, int axis, int radius, int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count)return;
  int stride=axis==0?n*n:(axis==1?n:1);
  int coordinate=(i/stride)%n;
  double value=weights[radius]*x[i];
  // Match scipy.ndimage's symmetric correlation accumulation order.
  for(int k=1;k<=radius;++k){
    int lo=(coordinate-k)%n;if(lo<0)lo+=n;
    int hi=(coordinate+k)%n;
    value+=weights[radius+k]*(x[i+int64_t(lo-coordinate)*stride]+x[i+int64_t(hi-coordinate)*stride]);
  }
  y[i]=value;
}
__global__ void project(const double* t,const double* beta,double* ph,double* dp,double* a,int64_t count){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  double b=beta[0],half=tanh(.5*b),q=tanh(b*(t[i]-.5));
  double p=(half+q)/(2*half);ph[i]=p;dp[i]=b*(1-q*q)/(2*half);
  a[i]=1e-4+(1.-1e-4)*p*p*p;
}
__global__ void make_candidate(const double* x,const double* ratio,const double* bounds,double* out,int64_t count){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  double lam=fmax((bounds[0]+bounds[1])*.5,1e-30);
  double lo=fmax(0.,x[i]-.1),hi=fmin(1.,x[i]+.1);
  out[i]=fmin(hi,fmax(lo,x[i]*sqrt(ratio[i]/lam)));
}
__global__ void update_bounds(const double* volume,double* bounds){
  double mid=(bounds[0]+bounds[1])*.5;
  if(volume[0]>.3)bounds[0]=mid;else bounds[1]=mid;
}
void check(const torch::Tensor& x){TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat64&&x.is_contiguous(),"contiguous CUDA FP64 required");}
void same(const torch::Tensor& a,const torch::Tensor& b){check(b);TORCH_CHECK(a.device()==b.device(),"device mismatch");}
}
void filter(torch::Tensor x,torch::Tensor weights,torch::Tensor y,int n,int axis){
  check(x);same(x,weights);same(x,y);
  TORCH_CHECK(n>=2&&axis>=0&&axis<3&&x.numel()==y.numel()&&x.numel()%(int64_t(n)*n*n)==0,"invalid filter shape");
  TORCH_CHECK(weights.numel()%2==1&&weights.numel()>=1&&weights.numel()<=4*n+1,"invalid Gaussian kernel");
  c10::cuda::CUDAGuard guard(x.device());
  gaussian_axis<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<double>(),weights.data_ptr<double>(),y.data_ptr<double>(),n,axis,weights.numel()/2,x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void physical(torch::Tensor t,torch::Tensor beta,torch::Tensor ph,torch::Tensor dp,torch::Tensor a){
  check(t);same(t,beta);same(t,ph);same(t,dp);same(t,a);
  TORCH_CHECK(beta.numel()==1&&ph.numel()==t.numel()&&dp.numel()==t.numel()&&a.numel()==t.numel(),"projection shape mismatch");
  c10::cuda::CUDAGuard guard(t.device());
  project<<<(t.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<double>(),beta.data_ptr<double>(),ph.data_ptr<double>(),dp.data_ptr<double>(),a.data_ptr<double>(),t.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void candidate(torch::Tensor x,torch::Tensor ratio,torch::Tensor bounds,torch::Tensor out){
  check(x);same(x,ratio);same(x,bounds);same(x,out);
  TORCH_CHECK(ratio.numel()==x.numel()&&out.numel()==x.numel()&&bounds.numel()==2,"OC shape mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  make_candidate<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<double>(),ratio.data_ptr<double>(),bounds.data_ptr<double>(),out.data_ptr<double>(),x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void bisect(torch::Tensor volume,torch::Tensor bounds){
  check(volume);same(volume,bounds);TORCH_CHECK(volume.numel()==1&&bounds.numel()==2,"OC scalar mismatch");
  c10::cuda::CUDAGuard guard(volume.device());
  update_bounds<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(volume.data_ptr<double>(),bounds.data_ptr<double>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("filter",&filter);m.def("physical",&physical);m.def("candidate",&candidate);m.def("bisect",&bisect);}
