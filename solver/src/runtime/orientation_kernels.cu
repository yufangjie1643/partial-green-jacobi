#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ int cx(int a){return (a==1||a==2||a==5||a==6);}
__device__ __forceinline__ int cy(int a){return (a==2||a==3||a==6||a==7);}
__device__ __forceinline__ int cz(int a){return a>=4;}
__device__ __forceinline__ int wrap(int i,int n){return i<0?i+n:(i>=n?i-n:i);}
__device__ __forceinline__ int index21(int i,int j){if(i>j){int t=i;i=j;j=t;}return 6*i-i*(i-1)/2+j-i;}

template<bool ORIENT>
__device__ __forceinline__ void constitutive(const double* C,const double* R,const int8_t* phase,const double* local,long n,const double* e,double* s){
  if constexpr(!ORIENT){
    #pragma unroll
    for(int i=0;i<6;i++){s[i]=0.;
      #pragma unroll
      for(int j=0;j<6;j++)s[i]+=C[n*21+index21(i,j)]*e[j];}
  }else{
    if(phase[n]==0){
      #pragma unroll
      for(int i=0;i<6;i++){s[i]=0.;
        #pragma unroll
        for(int j=0;j<6;j++)s[i]+=local[i*6+j]*e[j];}
      return;
    }
    double eg[9]={e[0],.5*e[5],.5*e[4],.5*e[5],e[1],.5*e[3],.5*e[4],.5*e[3],e[2]};
    double tmp[9],el[9],ev[6],sv[6],sl[9],sg[9];const double* r=R+n*9;
    #pragma unroll
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){tmp[i*3+j]=0.;for(int k=0;k<3;k++)tmp[i*3+j]+=eg[i*3+k]*r[k*3+j];}
    #pragma unroll
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){el[i*3+j]=0.;for(int k=0;k<3;k++)el[i*3+j]+=r[k*3+i]*tmp[k*3+j];}
    ev[0]=el[0];ev[1]=el[4];ev[2]=el[8];ev[3]=2*el[5];ev[4]=2*el[2];ev[5]=2*el[1];
    #pragma unroll
    for(int i=0;i<6;i++){sv[i]=0.;for(int j=0;j<6;j++)sv[i]+=local[36+i*6+j]*ev[j];}
    sl[0]=sv[0];sl[4]=sv[1];sl[8]=sv[2];sl[5]=sl[7]=sv[3];sl[2]=sl[6]=sv[4];sl[1]=sl[3]=sv[5];
    #pragma unroll
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){tmp[i*3+j]=0.;for(int k=0;k<3;k++)tmp[i*3+j]+=r[i*3+k]*sl[k*3+j];}
    #pragma unroll
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){sg[i*3+j]=0.;for(int k=0;k<3;k++)sg[i*3+j]+=tmp[i*3+k]*r[j*3+k];}
    s[0]=sg[0];s[1]=sg[4];s[2]=sg[8];s[3]=sg[5];s[4]=sg[2];s[5]=sg[1];
  }
}

template<bool ORIENT,int MODE>
__global__ void evaluate(const double* C,const double* R,const int8_t* phase,const double* local,const double* B,const double* HH,const double* hg,const double* u,const double* E,double* output,int nz,int ny,int nx,double V){
  long id=(long)blockIdx.x*blockDim.x+threadIdx.x,N=(long)nz*ny*nx;if(id>=N)return;
  int x=id%nx,y=(id/nx)%ny,z=id/(nx*ny);double force[3]={0.,0.,0.};
  int count=MODE==2?1:8;
  for(int a=0;a<count;a++){
    int ex=MODE==2?x:wrap(x-cx(a),nx),ey=MODE==2?y:wrap(y-cy(a),ny),ez=MODE==2?z:wrap(z-cz(a),nz);
    long el=((long)ez*ny+ey)*nx+ex;
    double ue[24],strain[6],stress[6];
    #pragma unroll
    for(int i=0;i<6;i++)strain[i]=MODE==0?0.:E[i];
    if constexpr(MODE!=1){
      #pragma unroll
      for(int b=0;b<8;b++){
        long node=((long)wrap(ez+cz(b),nz)*ny+wrap(ey+cy(b),ny))*nx+wrap(ex+cx(b),nx);
        #pragma unroll
        for(int p=0;p<3;p++)ue[b*3+p]=u[node*3+p];
      }
      #pragma unroll
      for(int i=0;i<6;i++)for(int b=0;b<24;b++)strain[i]+=B[i*24+b]*ue[b];
    }
    constitutive<ORIENT>(C,R,phase,local,el,strain,stress);
    if constexpr(MODE==2){
      #pragma unroll
      for(int i=0;i<6;i++)output[id*6+i]=stress[i];
    }else{
      #pragma unroll
      for(int p=0;p<3;p++){
        double f=0.;for(int i=0;i<6;i++)f+=B[i*24+a*3+p]*stress[i];f*=V;
        if constexpr(MODE==0){for(int b=0;b<8;b++)f+=hg[el]*HH[a*8+b]*ue[b*3+p];}
        force[p]+=(MODE==1?-f:f);
      }
    }
  }
  if constexpr(MODE!=2){for(int p=0;p<3;p++)output[id*3+p]=force[p];}
}

torch::Tensor run(torch::Tensor C,torch::Tensor R,torch::Tensor phase,torch::Tensor local,torch::Tensor B,torch::Tensor HH,torch::Tensor hg,torch::Tensor u,torch::Tensor E,int nz,int ny,int nx,double V,bool oriented,int mode){
  TORCH_CHECK(B.is_cuda() && B.scalar_type()==torch::kFloat64,"FP64 CUDA only");
  long N=(long)nz*ny*nx;auto out=torch::empty(mode==2?std::vector<int64_t>{1,N,6}:std::vector<int64_t>{1,3*N},B.options());
  auto stream=at::cuda::getCurrentCUDAStream();int blocks=(N+127)/128;
  #define LAUNCH(O,M) evaluate<O,M><<<blocks,128,0,stream>>>(C.data_ptr<double>(),R.data_ptr<double>(),phase.data_ptr<int8_t>(),local.data_ptr<double>(),B.data_ptr<double>(),HH.data_ptr<double>(),hg.data_ptr<double>(),u.data_ptr<double>(),E.data_ptr<double>(),out.data_ptr<double>(),nz,ny,nx,V)
  if(oriented){if(mode==0){LAUNCH(true,0);}else if(mode==1){LAUNCH(true,1);}else{LAUNCH(true,2);}}
  else{if(mode==0){LAUNCH(false,0);}else if(mode==1){LAUNCH(false,1);}else{LAUNCH(false,2);}}
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
