#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ long neighbor(long i,int a,int nz,int ny,int nx,int direction){
  int x=i%nx,y=(i/nx)%ny,z=i/(long(nx)*ny);
  x=(x+nx+direction*(a%4==1 || a%4==2))%nx;
  y=(y+ny+direction*(a%4>=2))%ny;
  z=(z+nz+direction*(a>=4))%nz;
  return (long(z)*ny+y)*nx+x;
}

template<int Mode,bool Atomic>
__global__ void local_action(const double* __restrict__ C,const double* __restrict__ G,
 const double* __restrict__ u,const double* __restrict__ macro,double* __restrict__ force,
 double* __restrict__ out,long n,int nz,int ny,int nx,double weight){
  long i=long(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  double ue[24],c[21],fe[24],mean[6]={0.,0.,0.,0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;a++){
    long node=neighbor(i,a,nz,ny,nx,1);
    #pragma unroll
    for(int p=0;p<3;p++){
      if(Mode!=1)ue[3*a+p]=u[3*node+p];
      fe[3*a+p]=0.;
    }
  }
  #pragma unroll
  for(int k=0;k<21;k++)c[k]=C[21*i+k];
  #pragma unroll 1
  for(int g=0;g<8;g++){
    double e[6]={0.,0.,0.,0.,0.,0.};
    if(Mode!=0){
      #pragma unroll
      for(int p=0;p<6;p++)e[p]=macro[p];
    }
    if(Mode!=1){
      #pragma unroll
      for(int a=0;a<8;a++){
        double dx=G[g*24+3*a],dy=G[g*24+3*a+1],dz=G[g*24+3*a+2];
        double ux=ue[3*a],uy=ue[3*a+1],uz=ue[3*a+2];
        e[0]+=dx*ux;e[1]+=dy*uy;e[2]+=dz*uz;
        e[3]+=dz*uy+dy*uz;e[4]+=dz*ux+dx*uz;e[5]+=dy*ux+dx*uy;
      }
    }
    double s[6]={0.,0.,0.,0.,0.,0.};int k=0;
    #pragma unroll
    for(int a=0;a<6;a++){
      #pragma unroll
      for(int b=a;b<6;b++){double v=c[k++];s[a]+=v*e[b];if(b!=a)s[b]+=v*e[a];}
    }
    if(Mode==2){
      #pragma unroll
      for(int p=0;p<6;p++)mean[p]+=s[p]/8.;
    }else{
      #pragma unroll
      for(int a=0;a<8;a++){
        double dx=G[g*24+3*a],dy=G[g*24+3*a+1],dz=G[g*24+3*a+2];
        fe[3*a]+=dx*s[0]+dz*s[4]+dy*s[5];
        fe[3*a+1]+=dy*s[1]+dz*s[3]+dx*s[5];
        fe[3*a+2]+=dz*s[2]+dy*s[3]+dx*s[4];
      }
    }
  }
  if(Mode==2){
    #pragma unroll
    for(int p=0;p<6;p++)out[i*6+p]=mean[p];
  }else{
    #pragma unroll
    for(int a=0;a<8;a++){
      long node=neighbor(i,a,nz,ny,nx,1);
      #pragma unroll
      for(int p=0;p<3;p++){
        double value=weight*(Mode==1?-1.:1.)*fe[3*a+p];
        if(Atomic)atomicAdd(out+3*node+p,value);
        else force[(3*a+p)*n+i]=value;
      }
    }
  }
}

__global__ void gather(const double* __restrict__ force,double* __restrict__ out,
 long n,int nz,int ny,int nx){
  long i=long(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  double sum[3]={0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;a++){
    long e=neighbor(i,a,nz,ny,nx,-1);
    #pragma unroll
    for(int p=0;p<3;p++)sum[p]+=force[(3*a+p)*n+e];
  }
  #pragma unroll
  for(int p=0;p<3;p++)out[3*i+p]=sum[p];
}

torch::Tensor evaluate(torch::Tensor C,torch::Tensor G,torch::Tensor u,torch::Tensor macro,
 torch::Tensor force,int nz,int ny,int nx,double weight,int mode,bool atomic,int threads){
  c10::cuda::CUDAGuard guard(C.device());
  for(auto t:{C,G,u,macro,force})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kDouble&&t.device()==C.device(),"contiguous same-device FP64 required");
  long n=long(nz)*ny*nx;
  TORCH_CHECK(nz>=2 && ny>=2 && nx>=2 && mode>=0 && mode<=2 && C.numel()==21*n && G.numel()==192,"dimensions");
  TORCH_CHECK(mode==1 || u.numel()==3*n,"displacement shape");
  TORCH_CHECK(mode==0 || macro.numel()==6,"macro shape");
  TORCH_CHECK(atomic || force.numel()==24*n,"force workspace");
  TORCH_CHECK(threads==64 || threads==128 || threads==256,"block size");
  auto out=mode==2?torch::empty({n,6},C.options()):
      (atomic?torch::zeros({1,3*n},C.options()):torch::empty({1,3*n},C.options()));
  auto stream=at::cuda::getCurrentCUDAStream();long blocks=(n+threads-1)/threads;
  #define CALL(M,A) local_action<M,A><<<blocks,threads,0,stream>>>(C.data_ptr<double>(),G.data_ptr<double>(),u.data_ptr<double>(),macro.data_ptr<double>(),force.data_ptr<double>(),out.data_ptr<double>(),n,nz,ny,nx,weight)
  if(mode==2){CALL(2,false);}
  else if(mode==1){if(atomic){CALL(1,true);}else{CALL(1,false);}}
  else {if(atomic){CALL(0,true);}else{CALL(0,false);}}
  if(mode!=2 && !atomic)gather<<<(n+127)/128,128,0,stream>>>(force.data_ptr<double>(),out.data_ptr<double>(),n,nz,ny,nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("evaluate",&evaluate);}
