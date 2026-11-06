#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Experimental FP64 kernels. C uses the 21 upper-triangular coefficients;
// B uses the original eight-point FI geometry, engineering shear 23,13,12.
__device__ __forceinline__ int ox(int a){return (a%4==1 || a%4==2);}
__device__ __forceinline__ int oy(int a){return a%4>=2;}
__device__ __forceinline__ int oz(int a){return a>=4;}
__device__ __forceinline__ long neighbor(long i,int a,int sign,int nz,int ny,int nx){
  int x=i%nx,y=(i/nx)%ny,z=i/(long(nx)*ny);
  x=(x+sign*ox(a)+nx)%nx; y=(y+sign*oy(a)+ny)%ny; z=(z+sign*oz(a)+nz)%nz;
  return (long(z)*ny+y)*nx+x;
}
__device__ __forceinline__ void cmul(const double* c,const double* e,double* s){
  #pragma unroll
  for(int j=0;j<6;j++) s[j]=0.;
  int k=0;
  #pragma unroll
  for(int i=0;i<6;i++){
    #pragma unroll
    for(int j=i;j<6;j++){
      double v=c[k++];s[i]+=v*e[j];if(i!=j)s[j]+=v*e[i];
    }
  }
}
__global__ void rhs_gather(const double* C,const double* Bint,const double* E,double* out,long n,int batch,int nz,int ny,int nx){
  long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
  long b=t/n,i=t%n;double f[3]={0.,0.,0.},e[6];
  #pragma unroll
  for(int k=0;k<6;k++)e[k]=E[k];
  #pragma unroll
  for(int a=0;a<8;a++){
    long el=b*n+neighbor(i,a,-1,nz,ny,nx);double s[6];cmul(C+21*el,e,s);
    #pragma unroll
    for(int p=0;p<3;p++){
      #pragma unroll
      for(int k=0;k<6;k++)f[p]-=Bint[k*24+3*a+p]*s[k];
    }
  }
  for(int p=0;p<3;p++)out[3*t+p]=f[p];
}
__global__ void stress_gp(const double* C,const double* B,const double* u,double* stress,long n,int batch,int nz,int ny,int nx){
  long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
  long b=t/n,i=t%n;double e[6]={0.,0.,0.,0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;a++){
    long j=b*n+neighbor(i,a,1,nz,ny,nx);
    double ux=u[3*j],uy=u[3*j+1],uz=u[3*j+2];
    double dx=B[3*a],dy=B[24+3*a+1],dz=B[48+3*a+2];
    e[0]+=dx*ux;e[1]+=dy*uy;e[2]+=dz*uz;
    e[3]+=dz*uy+dy*uz;e[4]+=dz*ux+dx*uz;e[5]+=dy*ux+dx*uy;
  }
  double s[6];cmul(C+21*t,e,s);
  #pragma unroll
  for(int k=0;k<6;k++)stress[k*n*batch+t]=s[k];
}
__global__ void force_gp(const double* stress,const double* B,double* out,long n,int batch,int nz,int ny,int nx,double weight,bool first){
  long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
  long b=t/n,i=t%n;double f[3]={0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;a++){
    long j=b*n+neighbor(i,a,-1,nz,ny,nx);double s[6];
    #pragma unroll
    for(int k=0;k<6;k++)s[k]=stress[k*n*batch+j];
    double dx=B[3*a],dy=B[24+3*a+1],dz=B[48+3*a+2];
    f[0]+=dx*s[0]+dz*s[4]+dy*s[5];
    f[1]+=dy*s[1]+dz*s[3]+dx*s[5];
    f[2]+=dz*s[2]+dy*s[3]+dx*s[4];
  }
  for(int p=0;p<3;p++)out[3*t+p]=(first?0.:out[3*t+p])+weight*f[p];
}
void check(torch::Tensor t){TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type()==at::kDouble,"contiguous CUDA FP64 required");}
torch::Tensor rhs(torch::Tensor C,torch::Tensor Bint,torch::Tensor E,int nz,int ny,int nx){
  check(C);check(Bint);check(E);c10::cuda::CUDAGuard guard(C.device());
  long n=long(nz)*ny*nx;int batch=C.size(0);
  auto out=torch::empty({batch,3*n},C.options());
  rhs_gather<<<(n*batch+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(C.data_ptr<double>(),Bint.data_ptr<double>(),E.data_ptr<double>(),out.data_ptr<double>(),n,batch,nz,ny,nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor direct(torch::Tensor C,torch::Tensor B,torch::Tensor u,torch::Tensor workspace,int nz,int ny,int nx,double detJ){
  check(C);check(B);check(u);check(workspace);c10::cuda::CUDAGuard guard(C.device());
  long n=long(nz)*ny*nx;int batch=C.size(0);TORCH_CHECK(workspace.numel()==6*n*batch,"stress workspace size");
  auto out=torch::empty_like(u);auto stream=at::cuda::getCurrentCUDAStream();long blocks=(n*batch+127)/128;
  for(int g=0;g<8;g++){
    stress_gp<<<blocks,128,0,stream>>>(C.data_ptr<double>(),B.data_ptr<double>()+g*144,u.data_ptr<double>(),workspace.data_ptr<double>(),n,batch,nz,ny,nx);
    force_gp<<<blocks,128,0,stream>>>(workspace.data_ptr<double>(),B.data_ptr<double>()+g*144,out.data_ptr<double>(),n,batch,nz,ny,nx,detJ,g==0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("rhs",&rhs);m.def("direct",&direct);}
