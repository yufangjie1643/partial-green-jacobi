#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
__device__ __forceinline__ int ox(int a){return a%4==1 || a%4==2;}
__device__ __forceinline__ int oy(int a){return a%4>=2;}
__device__ __forceinline__ long neighbor(long i,int a,int sign,int nz,int ny,int nx){
 int x=i%nx,y=(i/nx)%ny,z=i/(long(nx)*ny);
 x=(x+sign*ox(a)+nx)%nx;y=(y+sign*oy(a)+ny)%ny;z=(z+sign*(a>=4)+nz)%nz;
 return (long(z)*ny+y)*nx+x;
}
template<int Q> __global__ void stress(const double* C,const double* B,const double* u,double* work,long n,int batch,int nz,int ny,int nx){
 long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
 long b=t/n,i=t%n;double ue[24],c[21];
 #pragma unroll
 for(int a=0;a<8;a++){
  long j=b*n+neighbor(i,a,1,nz,ny,nx);
  #pragma unroll
  for(int p=0;p<3;p++)ue[3*a+p]=u[3*j+p];
 }
 #pragma unroll
 for(int k=0;k<21;k++)c[k]=C[21*t+k];
 for(int g=0;g<Q;g++){
  double e[6]={0.,0.,0.,0.,0.,0.};const double* bg=B+144*g;
  #pragma unroll
  for(int a=0;a<8;a++){
   double dx=bg[3*a],dy=bg[24+3*a+1],dz=bg[48+3*a+2];
   double ux=ue[3*a],uy=ue[3*a+1],uz=ue[3*a+2];
   e[0]+=dx*ux;e[1]+=dy*uy;e[2]+=dz*uz;e[3]+=dz*uy+dy*uz;e[4]+=dz*ux+dx*uz;e[5]+=dy*ux+dx*uy;
  }
  double s[6]={0.,0.,0.,0.,0.,0.};int k=0;
  #pragma unroll
  for(int a=0;a<6;a++){
   #pragma unroll
   for(int j=a;j<6;j++){double v=c[k++];s[a]+=v*e[j];if(j!=a)s[j]+=v*e[a];}
  }
  #pragma unroll
  for(int k=0;k<6;k++)work[(g*6+k)*n*batch+t]=s[k];
 }
}
template<int Q> __global__ void force(const double* work,const double* B,double* out,long n,int batch,int nz,int ny,int nx,double weight,bool first){
 long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
 long b=t/n,i=t%n;double f[3]={0.,0.,0.};
 #pragma unroll
 for(int a=0;a<8;a++){
  long j=b*n+neighbor(i,a,-1,nz,ny,nx);
  for(int g=0;g<Q;g++){
   const double* bg=B+144*g;double s[6];
   #pragma unroll
   for(int k=0;k<6;k++)s[k]=work[(g*6+k)*n*batch+j];
   double dx=bg[3*a],dy=bg[24+3*a+1],dz=bg[48+3*a+2];
   f[0]+=dx*s[0]+dz*s[4]+dy*s[5];f[1]+=dy*s[1]+dz*s[3]+dx*s[5];f[2]+=dz*s[2]+dy*s[3]+dx*s[4];
  }
 }
 for(int p=0;p<3;p++)out[3*t+p]=(first?0.:out[3*t+p])+weight*f[p];
}
template<int Q> void launch(torch::Tensor C,torch::Tensor B,torch::Tensor u,torch::Tensor work,torch::Tensor out,int nz,int ny,int nx,double w){
 long n=long(nz)*ny*nx;int batch=C.size(0);auto stream=at::cuda::getCurrentCUDAStream();long blocks=(n*batch+127)/128;
 for(int g=0;g<8;g+=Q){
  stress<Q><<<blocks,128,0,stream>>>(C.data_ptr<double>(),B.data_ptr<double>()+144*g,u.data_ptr<double>(),work.data_ptr<double>(),n,batch,nz,ny,nx);
  force<Q><<<blocks,128,0,stream>>>(work.data_ptr<double>(),B.data_ptr<double>()+144*g,out.data_ptr<double>(),n,batch,nz,ny,nx,w,g==0);
 }
}
torch::Tensor apply(torch::Tensor C,torch::Tensor B,torch::Tensor u,torch::Tensor work,int nz,int ny,int nx,double w,int q){
 c10::cuda::CUDAGuard guard(C.device());
 for(auto t:{C,B,u,work})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kDouble,"FP64 CUDA contiguous required");
 TORCH_CHECK(work.numel()==long(6)*q*C.size(0)*nz*ny*nx,"workspace size");
 auto out=torch::empty_like(u);
 if(q==2)launch<2>(C,B,u,work,out,nz,ny,nx,w);
 else if(q==4)launch<4>(C,B,u,work,out,nz,ny,nx,w);
 else if(q==8)launch<8>(C,B,u,work,out,nz,ny,nx,w);
 else TORCH_CHECK(false,"q must be 2,4,8");
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);}
