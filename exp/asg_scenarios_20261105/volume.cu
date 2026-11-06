#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void update(const double* volume, double* bounds, const double* target) {
  double mid = (bounds[0] + bounds[1]) * .5;
  if (volume[0] > target[0]) bounds[0] = mid;
  else bounds[1] = mid;
}
void bisect(torch::Tensor volume, torch::Tensor bounds, torch::Tensor target) {
  for (auto t : {volume, bounds, target}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),
                "contiguous CUDA FP64 required");
    TORCH_CHECK(t.device()==volume.device(), "device mismatch");
  }
  TORCH_CHECK(volume.numel()==1 && bounds.numel()==2 && target.numel()==1,
              "invalid scalar constraint");
  c10::cuda::CUDAGuard guard(volume.device());
  update<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(volume.data_ptr<double>(),
      bounds.data_ptr<double>(), target.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("bisect", &bisect); }
