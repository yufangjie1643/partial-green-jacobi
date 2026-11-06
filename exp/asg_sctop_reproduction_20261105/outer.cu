// Paper-matched clipped cone filter, tanh projection and damped OC update.
// The update follows SCTOP.m (Chao Li et al.; license preserved in vendor/).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
namespace {
__global__ void cone(const double* x,const int* offsets,const double* weights,
                     double* y,const double* bounds,int n,int m,bool conditional){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,N=int64_t(n)*n*n;
  if(i>=N || (conditional && bounds[2]==0))return;
  int xx=i%n,yy=i/n%n,zz=i/(n*n);double sum=0;
  for(int j=0;j<m;++j){
    int xj=xx+offsets[3*j],yj=yy+offsets[3*j+1],zj=zz+offsets[3*j+2];
    if(xj>=0 && xj<n && yj>=0 && yj<n && zj>=0 && zj<n)
      sum+=weights[j]*x[(int64_t(zj)*n+yj)*n+xj];
  }
  y[i]=sum;
}
__global__ void physical(const double* t,const double* hs,double* ph,double* dp,double* a,
                         const double* bounds,int64_t N,double beta,double eta,double emin,bool conditional){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=N || (conditional && bounds[2]==0))return;
  double denominator=tanh(beta*eta)+tanh(beta*(1-eta));
  double q=tanh(beta*(t[i]/hs[i]-eta));
  double rho=(tanh(beta*eta)+q)/denominator;
  ph[i]=rho;dp[i]=beta*(1-q*q)/denominator;a[i]=emin+(1-emin)*rho*rho*rho;
}
__global__ void candidate(const double* x,const double* ratio,const double* bounds,double* y,int64_t N){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=N||bounds[2]==0)return;
  double mid=.5*(bounds[0]+bounds[1]);
  double v=x[i]*pow(fmax(1e-10,ratio[i]/mid),.3);
  y[i]=fmax(.001,fmax(x[i]-.1,fmin(1.,fmin(x[i]+.1,v))));
}
__global__ void bisect(const double* volume,double* b,double target){
  if(b[2]==0)return;
  double mid=.5*(b[0]+b[1]);if(volume[0]>target)b[0]=mid;else b[1]=mid;
  b[3]+=1.;b[2]=((b[1]-b[0])/(b[1]+b[0])>1e-4 && b[1]>1e-40)?1.:0.;
}
void check(torch::Tensor x){TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==torch::kFloat64,"CUDA contiguous FP64 expected");}
}
void filter(torch::Tensor x,torch::Tensor offsets,torch::Tensor weights,torch::Tensor y,torch::Tensor bounds,int n,bool conditional){
  check(x);check(weights);check(y);check(bounds);c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(offsets.is_cuda()&&offsets.is_contiguous()&&offsets.scalar_type()==torch::kInt32,"int32 offsets expected");
  TORCH_CHECK(x.numel()==int64_t(n)*n*n&&y.numel()==x.numel()&&offsets.numel()==3*weights.numel(),"filter shapes");
  cone<<<(x.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<double>(),offsets.data_ptr<int>(),weights.data_ptr<double>(),y.data_ptr<double>(),bounds.data_ptr<double>(),n,weights.numel(),conditional);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void project(torch::Tensor t,torch::Tensor hs,torch::Tensor ph,torch::Tensor dp,torch::Tensor a,torch::Tensor bounds,double beta,double eta,double emin,bool conditional){
  check(t);check(hs);check(ph);check(dp);check(a);check(bounds);c10::cuda::CUDAGuard guard(t.device());
  physical<<<(t.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<double>(),hs.data_ptr<double>(),ph.data_ptr<double>(),dp.data_ptr<double>(),a.data_ptr<double>(),bounds.data_ptr<double>(),t.numel(),beta,eta,emin,conditional);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void make_candidate(torch::Tensor x,torch::Tensor ratio,torch::Tensor bounds,torch::Tensor out){
  check(x);check(ratio);check(bounds);check(out);c10::cuda::CUDAGuard guard(x.device());
  candidate<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<double>(),ratio.data_ptr<double>(),bounds.data_ptr<double>(),out.data_ptr<double>(),x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void update_bounds(torch::Tensor volume,torch::Tensor bounds,double target){
  check(volume);check(bounds);c10::cuda::CUDAGuard guard(volume.device());
  bisect<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(volume.data_ptr<double>(),bounds.data_ptr<double>(),target);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("filter",&filter);m.def("project",&project);m.def("candidate",&make_candidate);m.def("bisect",&update_bounds);}
