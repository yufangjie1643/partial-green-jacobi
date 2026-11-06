#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cufft.h>
#include <algorithm>
#include <climits>

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

#if FEAD_FFT_DOUBLE
using complex_t = cufftDoubleComplex;
constexpr cufftType kForwardType = CUFFT_D2Z, kInverseType = CUFFT_Z2D;
#define execute_forward cufftExecD2Z
#define execute_inverse cufftExecZ2D
#else
using complex_t = cufftComplex;
constexpr cufftType kForwardType = CUFFT_R2C, kInverseType = CUFFT_C2R;
#define execute_forward cufftExecR2C
#define execute_inverse cufftExecC2R
#endif

namespace {
void cufft_check(cufftResult code){TORCH_CHECK(code==CUFFT_SUCCESS,"cuFFT status ",int(code));}
class NativeFFT {
  int nz_,ny_,nx_,n_,nc_,device_;
  cudaStream_t stream_;
  cufftHandle forward_=0,inverse_=0;
  torch::Tensor workspace_;
  void validate(const torch::Tensor& x,c10::ScalarType type,int64_t count){
    TORCH_CHECK(x.is_cuda()&&x.get_device()==device_&&x.is_contiguous()
               &&x.scalar_type()==type&&x.numel()==count,"invalid native FFT input");
    TORCH_CHECK(at::cuda::getCurrentCUDAStream(device_).stream()==stream_,
                "NativeFFT workspace is owned by its creation stream");
  }
 public:
  NativeFFT(int nz,int ny,int nx):nz_(nz),ny_(ny),nx_(nx){
    TORCH_CHECK(nz>=2&&ny>=2&&nx>=2&&int64_t(nz)*ny*nx<INT_MAX/3,"invalid native FFT shape");
    n_=nz*ny*nx;nc_=nz*ny*(nx/2+1);device_=c10::cuda::current_device();
    stream_=at::cuda::getCurrentCUDAStream(device_).stream();
    try{
      cufft_check(cufftCreate(&forward_));cufft_check(cufftCreate(&inverse_));
      cufft_check(cufftSetAutoAllocation(forward_,0));cufft_check(cufftSetAutoAllocation(inverse_,0));
      int dims[3]={nz,ny,nx},real[3]={nz,ny,nx},complex[3]={nz,ny,nx/2+1};
      size_t fw=0,bw=0;
      cufft_check(cufftMakePlanMany(forward_,3,dims,real,1,n_,complex,1,nc_,kForwardType,3,&fw));
      cufft_check(cufftMakePlanMany(inverse_,3,dims,complex,1,nc_,real,1,n_,kInverseType,3,&bw));
      workspace_=torch::empty({int64_t(std::max(fw,bw))},torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA,device_)));
      cufft_check(cufftSetWorkArea(forward_,workspace_.data_ptr()));
      cufft_check(cufftSetWorkArea(inverse_,workspace_.data_ptr()));
      cufft_check(cufftSetStream(forward_,stream_));cufft_check(cufftSetStream(inverse_,stream_));
    }catch(...){if(forward_)cufftDestroy(forward_);if(inverse_)cufftDestroy(inverse_);throw;}
  }
  ~NativeFFT(){
    c10::cuda::CUDAGuard guard(c10::Device(torch::kCUDA,device_));
    if(forward_)cufftDestroy(forward_);if(inverse_)cufftDestroy(inverse_);
  }
  torch::Tensor forward(torch::Tensor x){
    c10::cuda::CUDAGuard guard(x.device());validate(x,kRealType,int64_t(3)*n_);
    auto out=torch::empty({1,3,nz_,ny_,nx_/2+1},x.options().dtype(kComplexType));
    cufft_check(execute_forward(forward_,x.data_ptr<real_t>(),reinterpret_cast<complex_t*>(out.data_ptr<c10::complex<real_t>>())));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
  }
  torch::Tensor inverse_destructive(torch::Tensor x){
    c10::cuda::CUDAGuard guard(x.device());validate(x,kComplexType,int64_t(3)*nc_);
    auto out=torch::empty({1,3,nz_,ny_,nx_},x.options().dtype(kRealType));
    cufft_check(execute_inverse(inverse_,reinterpret_cast<complex_t*>(x.data_ptr<c10::complex<real_t>>()),out.data_ptr<real_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
  }
  int64_t workspace_bytes()const{return workspace_.numel();}
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  pybind11::class_<NativeFFT,std::shared_ptr<NativeFFT>>(m,"NativeFFT")
    .def(pybind11::init<int,int,int>()).def("forward",&NativeFFT::forward)
    .def("inverse_destructive",&NativeFFT::inverse_destructive)
    .def_property_readonly("workspace_bytes",&NativeFFT::workspace_bytes);
}
