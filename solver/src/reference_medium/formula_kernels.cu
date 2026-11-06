#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
#include "modal_functions.cuh"

__device__ __forceinline__ int packed(int i,int j) {
  if(i>j){int t=i;i=j;j=t;}return 6*i-i*(i-1)/2+j-i;
}
__device__ __forceinline__ int wrap(int x,int n){return x==n?0:x;}
__device__ __forceinline__ void reduce2(double a,double b,double* out) {
  for(int d=16;d>0;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);b+=__shfl_down_sync(0xffffffff,b,d);}
  __shared__ double sa[8],sb[8];
  if((threadIdx.x&31)==0){sa[threadIdx.x/32]=a;sb[threadIdx.x/32]=b;}
  __syncthreads();
  if(threadIdx.x==0){a=0;b=0;for(int i=0;i<blockDim.x/32;++i){a+=sa[i];b+=sb[i];}
    out[blockIdx.x]=a;out[gridDim.x+blockIdx.x]=b;}
}
struct Geometry {int nx,ny,nz;double hx,hy,hz;};
__device__ __forceinline__ void nodes(int64_t e,Geometry g,int64_t* ids) {
  int x=e%g.nx,y=(e/g.nx)%g.ny,z=e/(int64_t(g.nx)*g.ny);
  #pragma unroll
  for(int a=0;a<8;++a)ids[a]=(int64_t(wrap(z+((a>>2)&1),g.nz))*g.ny+wrap(y+((a>>1)&1),g.ny))*g.nx+wrap(x+(a&1),g.nx);
}

__global__ void contrast_kernel(const double* u,const double* c,const double* scales,const int32_t* active,
    double* f,int64_t count,Geometry g) {
  const int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
  const int64_t n=int64_t(g.nx)*g.ny*g.nz;int64_t ids[8];nodes(active[j],g,ids);
  double local[24],force[24];
  #pragma unroll
  for(int a=0;a<8;++a){
    #pragma unroll
    for(int d=0;d<3;++d)local[3*a+d]=u[d*n+ids[a]];
  }
  const double geometry[4]={1/(4*g.hx),1/(4*g.hy),1/(4*g.hz),g.hx*g.hy*g.hz};
  hg_eval<false>(local,c+21*j,scales[j],geometry,force);
  #pragma unroll
  for(int a=0;a<8;++a){
    #pragma unroll
    for(int d=0;d<3;++d)atomicAdd(f+d*n+ids[a],force[3*a+d]);
  }
}

__global__ void rhs_kernel(const double* c,const int32_t* active,const double* macro,
    double* out,int64_t count,Geometry g) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(e>=count)return;
  double s[6]={};
  #pragma unroll
  for(int i=0;i<6;++i){
    #pragma unroll
    for(int j=0;j<6;++j)s[i]+=c[e*21+packed(i,j)]*macro[j];
  }
  int64_t ids[8];nodes(active[e],g,ids);
  const double vol=g.hx*g.hy*g.hz;
  #pragma unroll
  for(int a=0;a<8;++a){
    double x=((a&1)?1.:-1.)/(4*g.hx),y=((a&2)?1.:-1.)/(4*g.hy),z=((a&4)?1.:-1.)/(4*g.hz);
    atomicAdd(out+3*ids[a],-vol*(x*s[0]+y*s[5]+z*s[4]));
    atomicAdd(out+3*ids[a]+1,-vol*(x*s[5]+y*s[1]+z*s[3]));
    atomicAdd(out+3*ids[a]+2,-vol*(x*s[4]+y*s[3]+z*s[2]));
  }
}

__global__ void stress_kernel(const double* u,const double* c,const int32_t* active,
    const double* macro,double* partial,int64_t count,Geometry g) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  double strain[6],s[6]={};
  #pragma unroll
  for(int i=0;i<6;++i)strain[i]=macro[i];
  if(e<count){
    int64_t ids[8];nodes(active[e],g,ids);
    #pragma unroll
    for(int a=0;a<8;++a){
      const double x=((a&1)?1.:-1.)/(4*g.hx),y=((a&2)?1.:-1.)/(4*g.hy),z=((a&4)?1.:-1.)/(4*g.hz);
      const double ux=u[3*ids[a]],uy=u[3*ids[a]+1],uz=u[3*ids[a]+2];
      strain[0]+=x*ux;strain[1]+=y*uy;strain[2]+=z*uz;
      strain[3]+=z*uy+y*uz;strain[4]+=z*ux+x*uz;strain[5]+=y*ux+x*uy;
    }
    #pragma unroll
    for(int i=0;i<6;++i){
      #pragma unroll
      for(int j=0;j<6;++j)s[i]+=c[21*e+packed(i,j)]*strain[j];
    }
  }
  __shared__ double shared[6*128];
  #pragma unroll
  for(int i=0;i<6;++i)shared[i*128+threadIdx.x]=s[i];
  __syncthreads();
  for(int d=64;d>0;d/=2){if(threadIdx.x<d){
    #pragma unroll
    for(int i=0;i<6;++i)shared[i*128+threadIdx.x]+=shared[i*128+threadIdx.x+d];
  }__syncthreads();}
  if(threadIdx.x==0){
    #pragma unroll
    for(int i=0;i<6;++i)partial[i*gridDim.x+blockIdx.x]=shared[i*128];
  }
}

struct Wave {double s[3],c[3],d[3];int64_t count;int x;double weight;};
__device__ __forceinline__ Wave wave(int64_t k,const double* tx,const double* ty,const double* tz,Geometry g) {
  Wave w;int ns=g.nx/2+1;w.count=int64_t(ns)*g.ny*g.nz;
  int x=k%ns,y=(k/ns)%g.ny,z=k/(int64_t(ns)*g.ny);w.x=x;
  w.weight=(x==0||(g.nx%2==0&&x==g.nx/2))?1.:2.;
  w.s[0]=tx[2*x];w.c[0]=tx[2*x+1];w.s[1]=ty[2*y];w.c[1]=ty[2*y+1];w.s[2]=tz[2*z];w.c[2]=tz[2*z+1];
  w.d[0]=2/g.hx*w.s[0]*w.c[1]*w.c[2];w.d[1]=2/g.hy*w.c[0]*w.s[1]*w.c[2];w.d[2]=2/g.hz*w.c[0]*w.c[1]*w.s[2];
  return w;
}
template<typename T>
__global__ void fi_coeff_kernel(T* out,const double* tx,const double* ty,const double* tz,
    Geometry g,double lam,double mu) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;
  if(k>=n)return;
  Wave w=wave(k,tx,ty,tz,g);double m[3],v[3],d[3],h[3]={g.hx,g.hy,g.hz};
  double t=g.hx*g.hy*g.hz,trace=0;
  #pragma unroll
  for(int i=0;i<3;++i){m[i]=w.c[i]*w.c[i]+w.s[i]*w.s[i]/3;t*=m[i];v[i]=2*w.s[i]*w.c[i]/(h[i]*m[i]);d[i]=4*pow(w.s[i],4)/(3*h[i]*h[i]*m[i]*m[i]);trace+=d[i]+v[i]*v[i];}
  trace*=t;double b=(lam+mu)*t,den=1.;
  #pragma unroll
  for(int i=0;i<3;++i){double inv=k==0?0.:1/(mu*trace+(lam+mu)*t*d[i]);out[i*n+k]=T(inv);den+=b*v[i]*v[i]*inv;}
  out[3*n+k]=k==0?T(0):T(b/den);
}
template<typename T>
__global__ void fi_apply_kernel(const T* coef,const T* vx,const T* vy,const T* vz,
    const c10::complex<T>* r,c10::complex<T>* out,Geometry g) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;if(k>=n)return;
  int ns=g.nx/2+1,x=k%ns,y=(k/ns)%g.ny,z=k/(int64_t(ns)*g.ny);
  T v[3]={vx[x],vy[y],vz[z]},d[3];c10::complex<T> q[3],sum(0,0);
  #pragma unroll
  for(int i=0;i<3;++i){d[i]=coef[i*n+k];q[i]=d[i]*r[i*n+k];sum+=v[i]*q[i];}
  #pragma unroll
  for(int i=0;i<3;++i)out[i*n+k]=q[i]-(coef[3*n+k]*d[i]*v[i])*sum;
}

__global__ void hg_coeff_kernel(double* out,const double* tx,const double* ty,const double* tz,
    Geometry g,double l0,double m0,double p0,double lb,double mb,double pb) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;if(k>=n)return;
  Wave w=wave(k,tx,ty,tz,g);double q=0;for(int i=0;i<3;++i)q+=w.d[i]*w.d[i];
  double sx=w.s[0],sy=w.s[1],sz=w.s[2],cx=w.c[0],cy=w.c[1],cz=w.c[2];
  double mode=sx*sx*sy*sy*cz*cz+sx*sx*cy*cy*sz*sz+cx*cx*sy*sy*sz*sz+sx*sx*sy*sy*sz*sz;
  double vol=g.hx*g.hy*g.hz,t0=vol*m0*q+8*p0*mode,L0=t0+vol*(l0+m0)*q;
  double tb=vol*mb*q+8*pb*mode,Lb=tb+vol*(lb+mb)*q;
  double wt=k==0?0.:1/sqrt(t0),wl=k==0?0.:1/sqrt(L0),dt=k==0?0.:tb/t0,dl=k==0?0.:Lb/L0;
  out[k]=wt;out[n+k]=wl-wt;out[2*n+k]=dt;out[3*n+k]=dl-dt;
}
__device__ __forceinline__ void unit(Wave& w){double n=0;for(int i=0;i<3;++i)n+=w.d[i]*w.d[i];n=sqrt(n);for(int i=0;i<3;++i)w.d[i]=n>1e-24?w.d[i]/n:0.;}
__device__ __forceinline__ double norm(c10::complex<double> z){return z.real()*z.real()+z.imag()*z.imag();}
__device__ __forceinline__ void project_value(const c10::complex<double>* in,int64_t k,int64_t n,Wave w,double a,double b,c10::complex<double>* z){
  c10::complex<double> dot(0,0);for(int i=0;i<3;++i)dot+=w.d[i]*in[i*n+k];for(int i=0;i<3;++i)z[i]=a*in[i*n+k]+(b*w.d[i])*dot;
}
__global__ void project_kernel(const double* c,const double* tx,const double* ty,const double* tz,
    const c10::complex<double>* in,c10::complex<double>* out,Geometry g,int mode) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;if(k>=n)return;
  Wave w=wave(k,tx,ty,tz,g);unit(w);double a=c[k],b=c[n+k];
  if(mode==1){a=c[2*n+k];b=c[3*n+k];}
  if(mode==2){double wt=c[k],wl=wt+c[n+k];a=k==0?0.:c[2*n+k]/(wt*wt);b=(k==0?0.:(c[2*n+k]+c[3*n+k])/(wl*wl))-a;}
  if(mode==3){double wt=c[k],wl=wt+c[n+k];a=k==0?0.:1/wt;b=(k==0?0.:1/wl)-a;}
  c10::complex<double> z[3];project_value(in,k,n,w,a,b,z);for(int i=0;i<3;++i)out[i*n+k]=z[i];
}
__global__ void combine_kernel(const double* c,const double* tx,const double* ty,const double* tz,
    const c10::complex<double>* y,const c10::complex<double>* f,c10::complex<double>* out,double* partial,Geometry g) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;double value=0.;
  if(k<n){Wave w=wave(k,tx,ty,tz,g);unit(w);c10::complex<double> a[3],b[3];
    project_value(y,k,n,w,c[2*n+k],c[3*n+k],a);project_value(f,k,n,w,c[k],c[n+k],b);
    for(int i=0;i<3;++i){auto z=a[i]+b[i];out[i*n+k]=z;auto q=y[i*n+k];value+=w.weight*(q.real()*z.real()+q.imag()*z.imag());}
  }reduce2(value,0.,partial);
}
template<bool update>
__global__ void state_kernel(const double* c,const double* tx,const double* ty,const double* tz,
    c10::complex<double>* y,c10::complex<double>* r,const c10::complex<double>* p,const c10::complex<double>* ap,
    const double* rho,const double* pap,double* partial,Geometry g) {
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(g.nx/2+1)*g.ny*g.nz;double sum=0.,physical=0.;
  if(k<n){Wave w=wave(k,tx,ty,tz,g);unit(w);c10::complex<double> dot(0,0);
    double alpha=0;if constexpr(update)alpha=rho[0]/pap[0];
    for(int i=0;i<3;++i){auto rr=r[i*n+k];if constexpr(update){y[i*n+k]+=alpha*p[i*n+k];rr-=alpha*ap[i*n+k];r[i*n+k]=rr;}sum+=norm(rr);dot+=w.d[i]*rr;}
    double wt=c[k],wl=wt+c[n+k],a=k==0?0.:1/(wt*wt),b=(k==0?0.:1/(wl*wl))-a;
    physical=w.weight*(a*sum+b*norm(dot));sum*=w.weight;
  }reduce2(sum,physical,partial);
}
__global__ void direction_kernel(c10::complex<double>* p,const c10::complex<double>* r,const double* next,const double* old,int64_t n){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]=r[i]+(next[0]/old[0])*p[i];
}
}

Geometry geometry(int nz,int ny,int nx,double hx,double hy,double hz){return {nx,ny,nz,hx,hy,hz};}
torch::Tensor contrast(torch::Tensor c,torch::Tensor scales,torch::Tensor active,torch::Tensor u,int nz,int ny,int nx,double hx,double hy,double hz){
  auto out=torch::zeros_like(u);int64_t count=active.numel();if(count)contrast_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),active.data_ptr<int32_t>(),out.data_ptr<double>(),count,geometry(nz,ny,nx,hx,hy,hz));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor rhs(torch::Tensor c,torch::Tensor active,torch::Tensor macro,int nz,int ny,int nx,double hx,double hy,double hz){
  auto out=torch::zeros({1,int64_t(nx)*ny*nz*3},c.options());int64_t count=active.numel();if(count)rhs_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(c.data_ptr<double>(),active.data_ptr<int32_t>(),macro.data_ptr<double>(),out.data_ptr<double>(),count,geometry(nz,ny,nx,hx,hy,hz));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor stress(torch::Tensor c,torch::Tensor active,torch::Tensor macro,torch::Tensor u,int nz,int ny,int nx,double hx,double hy,double hz){
  int64_t count=active.numel();auto out=torch::empty({6,(count+127)/128},c.options());if(count)stress_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(u.data_ptr<double>(),c.data_ptr<double>(),active.data_ptr<int32_t>(),macro.data_ptr<double>(),out.data_ptr<double>(),count,geometry(nz,ny,nx,hx,hy,hz));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor fi_coeff(torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,int nz,int ny,int nx,double hx,double hy,double hz,double lam,double mu,bool fp64){
  int64_t n=int64_t(nx/2+1)*ny*nz;auto out=torch::empty({4,n},tx.options().dtype(fp64?torch::kFloat64:torch::kFloat32));auto g=geometry(nz,ny,nx,hx,hy,hz);auto stream=at::cuda::getCurrentCUDAStream();
  if(fp64)fi_coeff_kernel<<<(n+255)/256,256,0,stream>>>(out.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),g,lam,mu);
  else fi_coeff_kernel<<<(n+255)/256,256,0,stream>>>(out.data_ptr<float>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),g,lam,mu);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor fi_apply(torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,torch::Tensor r,int nz,int ny,int nx,double hx,double hy,double hz){
  auto out=torch::empty_like(r);int64_t n=int64_t(nx/2+1)*ny*nz;auto g=geometry(nz,ny,nx,hx,hy,hz);auto stream=at::cuda::getCurrentCUDAStream();
  if(r.scalar_type()==torch::kComplexDouble)fi_apply_kernel<<<(n+255)/256,256,0,stream>>>(c.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),r.data_ptr<c10::complex<double>>(),out.data_ptr<c10::complex<double>>(),g);
  else fi_apply_kernel<<<(n+255)/256,256,0,stream>>>(c.data_ptr<float>(),tx.data_ptr<float>(),ty.data_ptr<float>(),tz.data_ptr<float>(),r.data_ptr<c10::complex<float>>(),out.data_ptr<c10::complex<float>>(),g);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor hg_coeff(torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,int nz,int ny,int nx,double hx,double hy,double hz,double l0,double m0,double p0,double lb,double mb,double pb){
  int64_t n=int64_t(nx/2+1)*ny*nz;auto out=torch::empty({4,n},tx.options());hg_coeff_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),geometry(nz,ny,nx,hx,hy,hz),l0,m0,p0,lb,mb,pb);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor project(torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,torch::Tensor r,int nz,int ny,int nx,double hx,double hy,double hz,int mode){
  auto out=torch::empty_like(r);int64_t n=int64_t(nx/2+1)*ny*nz;project_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(c.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),r.data_ptr<c10::complex<double>>(),out.data_ptr<c10::complex<double>>(),geometry(nz,ny,nx,hx,hy,hz),mode);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> combine(torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,torch::Tensor y,torch::Tensor f,int nz,int ny,int nx,double hx,double hy,double hz){
  auto out=torch::empty_like(y);int64_t n=int64_t(nx/2+1)*ny*nz;int blocks=(n+255)/256;auto partial=torch::empty({2,blocks},c.options());combine_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(c.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),y.data_ptr<c10::complex<double>>(),f.data_ptr<c10::complex<double>>(),out.data_ptr<c10::complex<double>>(),partial.data_ptr<double>(),geometry(nz,ny,nx,hx,hy,hz));C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,partial};
}
torch::Tensor update(torch::Tensor c,torch::Tensor tx,torch::Tensor ty,torch::Tensor tz,torch::Tensor y,torch::Tensor r,torch::Tensor p,torch::Tensor ap,torch::Tensor rho,torch::Tensor pap,int nz,int ny,int nx,double hx,double hy,double hz,bool change){
  int64_t n=int64_t(nx/2+1)*ny*nz;int blocks=(n+255)/256;auto partial=torch::empty({2,blocks},c.options());auto g=geometry(nz,ny,nx,hx,hy,hz);auto stream=at::cuda::getCurrentCUDAStream();
  if(change)state_kernel<true><<<blocks,256,0,stream>>>(c.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),y.data_ptr<c10::complex<double>>(),r.data_ptr<c10::complex<double>>(),p.data_ptr<c10::complex<double>>(),ap.data_ptr<c10::complex<double>>(),rho.data_ptr<double>(),pap.data_ptr<double>(),partial.data_ptr<double>(),g);
  else state_kernel<false><<<blocks,256,0,stream>>>(c.data_ptr<double>(),tx.data_ptr<double>(),ty.data_ptr<double>(),tz.data_ptr<double>(),nullptr,r.data_ptr<c10::complex<double>>(),nullptr,nullptr,nullptr,nullptr,partial.data_ptr<double>(),g);C10_CUDA_KERNEL_LAUNCH_CHECK();return partial;
}
void direction(torch::Tensor p,torch::Tensor r,torch::Tensor next,torch::Tensor old){int64_t n=p.numel();direction_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<c10::complex<double>>(),r.data_ptr<c10::complex<double>>(),next.data_ptr<double>(),old.data_ptr<double>(),n);C10_CUDA_KERNEL_LAUNCH_CHECK();}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("contrast",&contrast);m.def("rhs",&rhs);m.def("stress",&stress);m.def("fi_coeff",&fi_coeff);m.def("fi_apply",&fi_apply);m.def("hg_coeff",&hg_coeff);m.def("project",&project);m.def("combine",&combine);m.def("update",&update);m.def("direction",&direction);}
