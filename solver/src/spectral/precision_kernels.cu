#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <climits>
#include <cmath>

// Identical native FFT/dataflow kernels, compiled for either precision.
#if FEAD_FFT_DOUBLE
using real_t = double;
constexpr auto kRealType = torch::kFloat64;
constexpr auto kComplexType = torch::kComplexDouble;
#else
using real_t = float;
constexpr auto kRealType = torch::kFloat32;
constexpr auto kComplexType = torch::kComplexFloat;
#endif

namespace {
using C = c10::complex<real_t>;
__global__ void pack_kernel(const double* __restrict__ x,real_t* __restrict__ y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  y[i]=real_t(x[3*i]);y[n+i]=real_t(x[3*i+1]);y[2*n+i]=real_t(x[3*i+2]);
}
template<bool scale>
__global__ void unpack_kernel(const real_t* __restrict__ x,double* __restrict__ y,int n,real_t factor){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  #pragma unroll
  for(int d=0;d<3;++d){real_t v=x[d*n+i];if constexpr(scale)v*=factor;y[3*i+d]=double(v);}
}

template<bool fi>
__global__ void original_inplace(C* r,const real_t* coef,const double* tx,const double* ty,const double* tz,
                      const real_t* vx,const real_t* vy,const real_t* vz,
                      int nz,int ny,int nx,double hx,double hy,double hz){
  int k=blockIdx.x*blockDim.x+threadIdx.x,ns=nx/2+1,n=ns*ny*nz;if(k>=n)return;
  int x=k%ns,y=(k/ns)%ny,z=k/(ns*ny);
  C q[3]={r[k],r[n+k],r[2*n+k]};
  if constexpr(fi){
    real_t v[3]={vx[x],vy[y],vz[z]},a[3]={coef[k],coef[n+k],coef[2*n+k]};
    C s(0,0);
    #pragma unroll
    for(int d=0;d<3;++d){q[d]=a[d]*q[d];s+=v[d]*q[d];}
    #pragma unroll
    for(int d=0;d<3;++d)r[d*n+k]=q[d]-(coef[3*n+k]*a[d]*v[d])*s;
  }else{
    const real_t gx=real_t(2/hx*tx[2*x]*ty[2*y+1]*tz[2*z+1]);
    const real_t gy=real_t(2/hy*tx[2*x+1]*ty[2*y]*tz[2*z+1]);
    const real_t gz=real_t(2/hz*tx[2*x+1]*ty[2*y+1]*tz[2*z]);
    const C dot=gx*q[0]+gy*q[1]+gz*q[2];
    const real_t a=coef[k],b=coef[n+k];
    r[k]=a*(q[0]-(b*gx)*dot);r[n+k]=a*(q[1]-(b*gy)*dot);r[2*n+k]=a*(q[2]-(b*gz)*dot);
  }
}

template<bool fi,bool dense>
__global__ void prepare_kernel(const real_t* coef,const double* tx,const double* ty,const double* tz,
                      const real_t* vx,const real_t* vy,const real_t* vz,real_t* out,
                      int nz,int ny,int nx,double hx,double hy,double hz){
  int k=blockIdx.x*blockDim.x+threadIdx.x,ns=nx/2+1,n=ns*ny*nz;if(k>=n)return;
  int x=k%ns,y=(k/ns)%ny,z=k/(ns*ny);
  double a[3],v[3],beta,invN=1./(double(nx)*ny*nz);
  if constexpr(fi){
    a[0]=coef[k];a[1]=coef[n+k];a[2]=coef[2*n+k];beta=coef[3*n+k];
    v[0]=a[0]*vx[x];v[1]=a[1]*vy[y];v[2]=a[2]*vz[z];
  }else{
    a[0]=a[1]=a[2]=coef[k];beta=double(coef[k])*coef[n+k];
    // Match the old FP64 geometry products followed by FP32 direction cast.
    v[0]=real_t(2/hx*tx[2*x]*ty[2*y+1]*tz[2*z+1]);
    v[1]=real_t(2/hy*tx[2*x+1]*ty[2*y]*tz[2*z+1]);
    v[2]=real_t(2/hz*tx[2*x+1]*ty[2*y+1]*tz[2*z]);
  }
  if constexpr(dense){
    out[k]=real_t((a[0]-beta*v[0]*v[0])*invN);
    out[n+k]=real_t((a[1]-beta*v[1]*v[1])*invN);
    out[2*n+k]=real_t((a[2]-beta*v[2]*v[2])*invN);
    out[3*n+k]=real_t(-beta*v[0]*v[1]*invN);
    out[4*n+k]=real_t(-beta*v[0]*v[2]*invN);
    out[5*n+k]=real_t(-beta*v[1]*v[2]*invN);
  }else{
    constexpr int offset=fi?3:1;
    out[k]=real_t(a[0]*invN);
    if constexpr(fi){out[n+k]=real_t(a[1]*invN);out[2*n+k]=real_t(a[2]*invN);}
    double root=sqrt(fmax(beta*invN,0.));
    #pragma unroll
    for(int d=0;d<3;++d)out[(offset+d)*n+k]=real_t(root*v[d]);
  }
}

template<bool fi,bool dense>
__global__ void cached_inplace(C* r,const real_t* __restrict__ c,int n){
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
  C x=r[k],y=r[n+k],z=r[2*n+k];
  if constexpr(dense){
    real_t c0=c[k],c1=c[n+k],c2=c[2*n+k],c3=c[3*n+k],c4=c[4*n+k],c5=c[5*n+k];
    r[k]=c0*x+c3*y+c4*z;r[n+k]=c3*x+c1*y+c5*z;r[2*n+k]=c4*x+c5*y+c2*z;
  }else{
    constexpr int offset=fi?3:1;
    real_t w0=c[offset*n+k],w1=c[(offset+1)*n+k],w2=c[(offset+2)*n+k];
    C s=w0*x+w1*y+w2*z;
    real_t a0=c[k],a1=fi?c[n+k]:a0,a2=fi?c[2*n+k]:a0;
    r[k]=a0*x-w0*s;r[n+k]=a1*y-w1*s;r[2*n+k]=a2*z-w2*s;
  }
}

void check(const torch::Tensor& t,c10::ScalarType dtype){
  TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==dtype,"contiguous CUDA tensor with expected dtype required");
}
int count(int nz,int ny,int nx){
  TORCH_CHECK(nz>=2&&ny>=2&&nx>=2&&int64_t(nz)*ny*nx<INT_MAX/6,"invalid shape/index range");return nz*ny*nx;
}
}

torch::Tensor pack(torch::Tensor x,int nz,int ny,int nx){
  check(x,torch::kFloat64);int n=count(nz,ny,nx);TORCH_CHECK(x.numel()==3*n,"B=1 required");
  c10::cuda::CUDAGuard guard(x.device());auto out=torch::empty({1,3,nz,ny,nx},x.options().dtype(kRealType));
  pack_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<double>(),out.data_ptr<real_t>(),n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor unpack(torch::Tensor x,int nz,int ny,int nx,bool normalized){
  check(x,kRealType);int n=count(nz,ny,nx);TORCH_CHECK(x.numel()==3*n,"invalid input shape");
  c10::cuda::CUDAGuard guard(x.device());auto out=torch::empty({1,3*n},x.options().dtype(torch::kFloat64));
  auto stream=at::cuda::getCurrentCUDAStream();
  if(normalized)unpack_kernel<true><<<(n+255)/256,256,0,stream>>>(x.data_ptr<real_t>(),out.data_ptr<double>(),n,real_t(1./n));
  else unpack_kernel<false><<<(n+255)/256,256,0,stream>>>(x.data_ptr<real_t>(),out.data_ptr<double>(),n,1.f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor prepare(torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,
                     int nz,int ny,int nx,double hx,double hy,double hz,bool fi,bool dense){
  check(c,kRealType);count(nz,ny,nx);int n=(nx/2+1)*ny*nz;
  TORCH_CHECK(c.numel()==(fi?4:2)*n,"invalid original coefficient shape");
  check(tx,fi?kRealType:torch::kFloat64);check(ty,tx.scalar_type());check(tz,tx.scalar_type());
  TORCH_CHECK(c.device()==tx.device()&&c.device()==ty.device()&&c.device()==tz.device(),"device mismatch");
  TORCH_CHECK(tx.numel()==(nx/2+1)*(fi?1:2)&&ty.numel()==ny*(fi?1:2)&&tz.numel()==nz*(fi?1:2),"invalid direction tables");
  c10::cuda::CUDAGuard guard(c.device());auto out=torch::empty({dense||fi?6:4,n},c.options());auto s=at::cuda::getCurrentCUDAStream();
  #define PREP(F,D) prepare_kernel<F,D><<<(n+255)/256,256,0,s>>>(c.data_ptr<real_t>(),F?nullptr:tx.data_ptr<double>(),F?nullptr:ty.data_ptr<double>(),F?nullptr:tz.data_ptr<double>(),F?tx.data_ptr<real_t>():nullptr,F?ty.data_ptr<real_t>():nullptr,F?tz.data_ptr<real_t>():nullptr,out.data_ptr<real_t>(),nz,ny,nx,hx,hy,hz)
  if(fi){if(dense){PREP(true,true);}else{PREP(true,false);}}
  else{if(dense){PREP(false,true);}else{PREP(false,false);}}
  #undef PREP
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void spectral(torch::Tensor r,torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,
              int nz,int ny,int nx,double hx,double hy,double hz,bool fi,int mode){
  check(r,kComplexType);check(c,kRealType);count(nz,ny,nx);int n=(nx/2+1)*ny*nz;
  TORCH_CHECK(r.numel()==3*n&&r.device()==c.device(),"invalid spectrum");
  TORCH_CHECK(mode>=0&&mode<=2,"invalid mode");
  TORCH_CHECK(c.numel()==(mode==0?(fi?4:2):(mode==1?(fi?6:4):6))*n,"invalid cached coefficients");
  c10::cuda::CUDAGuard guard(r.device());auto s=at::cuda::getCurrentCUDAStream();
  if(mode==0){
    #define ORIG(F) original_inplace<F><<<(n+255)/256,256,0,s>>>(r.data_ptr<C>(),c.data_ptr<real_t>(),F?nullptr:tx.data_ptr<double>(),F?nullptr:ty.data_ptr<double>(),F?nullptr:tz.data_ptr<double>(),F?tx.data_ptr<real_t>():nullptr,F?ty.data_ptr<real_t>():nullptr,F?tz.data_ptr<real_t>():nullptr,nz,ny,nx,hx,hy,hz)
    if(fi){ORIG(true);}else{ORIG(false);}
    #undef ORIG
  }else{
    #define CACHED(F,D) cached_inplace<F,D><<<(n+255)/256,256,0,s>>>(r.data_ptr<C>(),c.data_ptr<real_t>(),n)
    if(fi){if(mode==2){CACHED(true,true);}else{CACHED(true,false);}}
    else{if(mode==2){CACHED(false,true);}else{CACHED(false,false);}}
    #undef CACHED
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pack",&pack);m.def("unpack",&unpack);m.def("prepare",&prepare);m.def("spectral",&spectral);}
