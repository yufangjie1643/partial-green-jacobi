#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// One thread computes all eight FI Gauss points for one voxel. Only after
// summing the element's 24 force entries do we scatter using FP64 atomicAdd.
__device__ __forceinline__ long node(long i,int a,int nz,int ny,int nx){
  int x=i%nx,y=(i/nx)%ny,z=i/(long(nx)*ny);
  x=(x+(a%4==1 || a%4==2))%nx;y=(y+(a%4>=2))%ny;z=(z+(a>=4))%nz;
  return (long(z)*ny+y)*nx+x;
}
__global__ void element_atomic(const double* __restrict__ C,const double* __restrict__ B,
                              const double* __restrict__ u,double* __restrict__ out,
                              long n,int batch,int nz,int ny,int nx,double weight){
  long t=long(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=n*batch)return;
  long b=t/n,i=t%n;long nodes[8];double ue[24],c[21],fe[24];
  #pragma unroll
  for(int a=0;a<8;a++){
    nodes[a]=b*n+node(i,a,nz,ny,nx);
    #pragma unroll
    for(int p=0;p<3;p++){ue[3*a+p]=u[3*nodes[a]+p];fe[3*a+p]=0.;}
  }
  #pragma unroll
  for(int k=0;k<21;k++)c[k]=C[21*t+k];
  #pragma unroll 1
  for(int g=0;g<8;g++){
    const double* bg=B+144*g;double e[6]={0.,0.,0.,0.,0.,0.};
    #pragma unroll
    for(int a=0;a<8;a++){
      double dx=bg[3*a],dy=bg[24+3*a+1],dz=bg[48+3*a+2];
      double ux=ue[3*a],uy=ue[3*a+1],uz=ue[3*a+2];
      e[0]+=dx*ux;e[1]+=dy*uy;e[2]+=dz*uz;
      e[3]+=dz*uy+dy*uz;e[4]+=dz*ux+dx*uz;e[5]+=dy*ux+dx*uy;
    }
    double s[6]={0.,0.,0.,0.,0.,0.};int k=0;
    #pragma unroll
    for(int a=0;a<6;a++){
      #pragma unroll
      for(int j=a;j<6;j++){double v=c[k++];s[a]+=v*e[j];if(j!=a)s[j]+=v*e[a];}
    }
    #pragma unroll
    for(int a=0;a<8;a++){
      double dx=bg[3*a],dy=bg[24+3*a+1],dz=bg[48+3*a+2];
      fe[3*a]+=dx*s[0]+dz*s[4]+dy*s[5];
      fe[3*a+1]+=dy*s[1]+dz*s[3]+dx*s[5];
      fe[3*a+2]+=dz*s[2]+dy*s[3]+dx*s[4];
    }
  }
  #pragma unroll
  for(int a=0;a<8;a++){
    #pragma unroll
    for(int p=0;p<3;p++)atomicAdd(out+3*nodes[a]+p,weight*fe[3*a+p]);
  }
}
torch::Tensor apply(torch::Tensor C,torch::Tensor B,torch::Tensor u,int nz,int ny,int nx,double weight,int threads){
  TORCH_CHECK(threads==64 || threads==128 || threads==256,"block must be 64,128,256");
  c10::cuda::CUDAGuard guard(C.device());
  for(auto t:{C,B,u})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kDouble&&t.device()==C.device(),"same-device contiguous CUDA FP64 required");
  long n=long(nz)*ny*nx;int batch=C.size(0);
  TORCH_CHECK(C.numel()==21*n*batch && B.numel()==8*6*24 && u.numel()==3*n*batch,"tensor shapes");
  auto out=torch::zeros_like(u);
  element_atomic<<<(n*batch+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream()>>>(C.data_ptr<double>(),B.data_ptr<double>(),u.data_ptr<double>(),out.data_ptr<double>(),n,batch,nz,ny,nx,weight);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
pybind11::dict attributes(){
  cudaFuncAttributes attr;C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,element_atomic));
  pybind11::dict d;d["num_registers"]=attr.numRegs;d["local_bytes"]=attr.localSizeBytes;
  d["max_threads_per_block"]=attr.maxThreadsPerBlock;return d;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("attributes",&attributes);}
