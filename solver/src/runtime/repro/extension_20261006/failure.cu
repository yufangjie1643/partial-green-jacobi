#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// No fast-math: criterion and loads use FP64 throughout. One block reduces a
// bounded voxel tile for one RVE/direction; no direction-by-voxel stress array.
__global__ void evaluate_kernel(const double* S,const bool* mask,const double* d,
 const double* p,double* out,int N,int D,int tiles){
 int b=blockIdx.z,j=blockIdx.y,tile=blockIdx.x,e=tile*blockDim.x+threadIdx.x;
 double fy=0.,fm=0.;
 if(e<N){
   double s[6];
   for(int i=0;i<6;i++){double v=0.;for(int k=0;k<6;k++)v+=S[((long long)b*N+e)*36+i*6+k]*d[j*6+k];s[i]=v;}
   if(mask[b*N+e]){
     for(int i=0;i<3;i++)fy=fmax(fy,s[i]>=0?s[i]/p[i]:-s[i]/p[i+3]);
     for(int i=3;i<6;i++)fy=fmax(fy,fabs(s[i])/p[i+3]);
   }else{
     double v=.5*((s[0]-s[1])*(s[0]-s[1])+(s[1]-s[2])*(s[1]-s[2])+(s[2]-s[0])*(s[2]-s[0]))
             +3*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);
     double a=v/(p[9]*p[10]),l=(1/p[9]-1/p[10])*(s[0]+s[1]+s[2]);
     double r=sqrt(l*l+4*a);
     fm=l>=0?.5*(r+l):(r-l>0?2*a/(r-l):0.);
   }
 }
 __shared__ double sy[128],sm[128];int t=threadIdx.x;sy[t]=fy;sm[t]=fm;__syncthreads();
 for(int stride=64;stride>0;stride/=2){if(t<stride){sy[t]=fmax(sy[t],sy[t+stride]);sm[t]=fmax(sm[t],sm[t+stride]);}__syncthreads();}
 if(t==0){out[(((long long)b*D+j)*2)*tiles+tile]=sy[0];out[(((long long)b*D+j)*2+1)*tiles+tile]=sm[0];}
}
torch::Tensor evaluate_cuda(torch::Tensor S,torch::Tensor mask,torch::Tensor d,torch::Tensor p){
 TORCH_CHECK(S.is_cuda()&&S.scalar_type()==torch::kFloat64&&S.is_contiguous(),"FP64 CUDA contiguous basis required");
 TORCH_CHECK(mask.is_cuda()&&mask.scalar_type()==torch::kBool&&mask.is_contiguous(),"CUDA boolean mask required");
 TORCH_CHECK(S.dim()==4&&S.size(2)==6&&S.size(3)==6&&d.dim()==2&&d.size(1)==6,"invalid basis or directions");
 TORCH_CHECK(d.is_cuda()&&d.scalar_type()==torch::kFloat64&&d.is_contiguous()&&p.is_cuda()&&p.numel()==11,"invalid parameters");
 int B=S.size(0),N=S.size(1),D=d.size(0),tiles=(N+127)/128;
 auto out=torch::empty({B,D,2,tiles},S.options());
 evaluate_kernel<<<dim3(tiles,D,B),128,0,at::cuda::getCurrentCUDAStream()>>>(S.data_ptr<double>(),mask.data_ptr<bool>(),d.data_ptr<double>(),p.data_ptr<double>(),out.data_ptr<double>(),N,D,tiles);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
