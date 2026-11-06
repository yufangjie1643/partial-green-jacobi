
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__device__ __forceinline__ void hg_modal(const double* u, const double* c, double scale, const double* g, double* f) {
  const double q0_0_0 = u[0] + u[3];
  const double q0_1_0 = u[3] - u[0];
  const double q0_0_1 = u[1] + u[4];
  const double q0_1_1 = u[4] - u[1];
  const double q0_0_2 = u[2] + u[5];
  const double q0_1_2 = u[5] - u[2];
  const double q0_2_0 = u[6] + u[9];
  const double q0_3_0 = u[9] - u[6];
  const double q0_2_1 = u[7] + u[10];
  const double q0_3_1 = u[10] - u[7];
  const double q0_2_2 = u[8] + u[11];
  const double q0_3_2 = u[11] - u[8];
  const double q0_4_0 = u[12] + u[15];
  const double q0_5_0 = u[15] - u[12];
  const double q0_4_1 = u[13] + u[16];
  const double q0_5_1 = u[16] - u[13];
  const double q0_4_2 = u[14] + u[17];
  const double q0_5_2 = u[17] - u[14];
  const double q0_6_0 = u[18] + u[21];
  const double q0_7_0 = u[21] - u[18];
  const double q0_6_1 = u[19] + u[22];
  const double q0_7_1 = u[22] - u[19];
  const double q0_6_2 = u[20] + u[23];
  const double q0_7_2 = u[23] - u[20];
  const double q1_0_0 = q0_0_0 + q0_2_0;
  const double q1_2_0 = q0_2_0 - q0_0_0;
  const double q1_0_1 = q0_0_1 + q0_2_1;
  const double q1_2_1 = q0_2_1 - q0_0_1;
  const double q1_0_2 = q0_0_2 + q0_2_2;
  const double q1_2_2 = q0_2_2 - q0_0_2;
  const double q1_1_0 = q0_1_0 + q0_3_0;
  const double q1_3_0 = q0_3_0 - q0_1_0;
  const double q1_1_1 = q0_1_1 + q0_3_1;
  const double q1_3_1 = q0_3_1 - q0_1_1;
  const double q1_1_2 = q0_1_2 + q0_3_2;
  const double q1_3_2 = q0_3_2 - q0_1_2;
  const double q1_4_0 = q0_4_0 + q0_6_0;
  const double q1_6_0 = q0_6_0 - q0_4_0;
  const double q1_4_1 = q0_4_1 + q0_6_1;
  const double q1_6_1 = q0_6_1 - q0_4_1;
  const double q1_4_2 = q0_4_2 + q0_6_2;
  const double q1_6_2 = q0_6_2 - q0_4_2;
  const double q1_5_0 = q0_5_0 + q0_7_0;
  const double q1_7_0 = q0_7_0 - q0_5_0;
  const double q1_5_1 = q0_5_1 + q0_7_1;
  const double q1_7_1 = q0_7_1 - q0_5_1;
  const double q1_5_2 = q0_5_2 + q0_7_2;
  const double q1_7_2 = q0_7_2 - q0_5_2;
  const double q2_0_0 = q1_0_0 + q1_4_0;
  const double q2_4_0 = q1_4_0 - q1_0_0;
  const double q2_0_1 = q1_0_1 + q1_4_1;
  const double q2_4_1 = q1_4_1 - q1_0_1;
  const double q2_0_2 = q1_0_2 + q1_4_2;
  const double q2_4_2 = q1_4_2 - q1_0_2;
  const double q2_1_0 = q1_1_0 + q1_5_0;
  const double q2_5_0 = q1_5_0 - q1_1_0;
  const double q2_1_1 = q1_1_1 + q1_5_1;
  const double q2_5_1 = q1_5_1 - q1_1_1;
  const double q2_1_2 = q1_1_2 + q1_5_2;
  const double q2_5_2 = q1_5_2 - q1_1_2;
  const double q2_2_0 = q1_2_0 + q1_6_0;
  const double q2_6_0 = q1_6_0 - q1_2_0;
  const double q2_2_1 = q1_2_1 + q1_6_1;
  const double q2_6_1 = q1_6_1 - q1_2_1;
  const double q2_2_2 = q1_2_2 + q1_6_2;
  const double q2_6_2 = q1_6_2 - q1_2_2;
  const double q2_3_0 = q1_3_0 + q1_7_0;
  const double q2_7_0 = q1_7_0 - q1_3_0;
  const double q2_3_1 = q1_3_1 + q1_7_1;
  const double q2_7_1 = q1_7_1 - q1_3_1;
  const double q2_3_2 = q1_3_2 + q1_7_2;
  const double q2_7_2 = q1_7_2 - q1_3_2;
  const double e0_0 = g[0] * q2_1_0;
  const double e0_1 = g[1] * q2_2_1;
  const double e0_2 = g[2] * q2_4_2;
  const double e0_3 = g[2] * q2_4_1 + g[1] * q2_2_2;
  const double e0_4 = g[2] * q2_4_0 + g[0] * q2_1_2;
  const double e0_5 = g[1] * q2_2_0 + g[0] * q2_1_1;
  const double s0_0 = (c[0] * e0_0 + c[1] * e0_1 + c[2] * e0_2 + c[3] * e0_3 + c[4] * e0_4 + c[5] * e0_5) * (g[3] / 1.0);
  const double s0_1 = (c[1] * e0_0 + c[6] * e0_1 + c[7] * e0_2 + c[8] * e0_3 + c[9] * e0_4 + c[10] * e0_5) * (g[3] / 1.0);
  const double s0_2 = (c[2] * e0_0 + c[7] * e0_1 + c[11] * e0_2 + c[12] * e0_3 + c[13] * e0_4 + c[14] * e0_5) * (g[3] / 1.0);
  const double s0_3 = (c[3] * e0_0 + c[8] * e0_1 + c[12] * e0_2 + c[15] * e0_3 + c[16] * e0_4 + c[17] * e0_5) * (g[3] / 1.0);
  const double s0_4 = (c[4] * e0_0 + c[9] * e0_1 + c[13] * e0_2 + c[16] * e0_3 + c[18] * e0_4 + c[19] * e0_5) * (g[3] / 1.0);
  const double s0_5 = (c[5] * e0_0 + c[10] * e0_1 + c[14] * e0_2 + c[17] * e0_3 + c[19] * e0_4 + c[20] * e0_5) * (g[3] / 1.0);
  const double p0_0 = 0.0;
  const double p0_1 = 0.0;
  const double p0_2 = 0.0;
  const double p1_0 = g[0] * s0_0;
  const double p1_1 = g[0] * s0_5;
  const double p1_2 = g[0] * s0_4;
  const double p2_0 = g[1] * s0_5;
  const double p2_1 = g[1] * s0_1;
  const double p2_2 = g[1] * s0_3;
  const double p3_0 = scale * 0.125 * q2_3_0;
  const double p3_1 = scale * 0.125 * q2_3_1;
  const double p3_2 = scale * 0.125 * q2_3_2;
  const double p4_0 = g[2] * s0_4;
  const double p4_1 = g[2] * s0_3;
  const double p4_2 = g[2] * s0_2;
  const double p5_0 = scale * 0.125 * q2_5_0;
  const double p5_1 = scale * 0.125 * q2_5_1;
  const double p5_2 = scale * 0.125 * q2_5_2;
  const double p6_0 = scale * 0.125 * q2_6_0;
  const double p6_1 = scale * 0.125 * q2_6_1;
  const double p6_2 = scale * 0.125 * q2_6_2;
  const double p7_0 = scale * 0.125 * q2_7_0;
  const double p7_1 = scale * 0.125 * q2_7_1;
  const double p7_2 = scale * 0.125 * q2_7_2;
  const double f0_0_0 = p0_0 - p1_0;
  const double f0_1_0 = p0_0 + p1_0;
  const double f0_0_1 = p0_1 - p1_1;
  const double f0_1_1 = p0_1 + p1_1;
  const double f0_0_2 = p0_2 - p1_2;
  const double f0_1_2 = p0_2 + p1_2;
  const double f0_2_0 = p2_0 - p3_0;
  const double f0_3_0 = p2_0 + p3_0;
  const double f0_2_1 = p2_1 - p3_1;
  const double f0_3_1 = p2_1 + p3_1;
  const double f0_2_2 = p2_2 - p3_2;
  const double f0_3_2 = p2_2 + p3_2;
  const double f0_4_0 = p4_0 - p5_0;
  const double f0_5_0 = p4_0 + p5_0;
  const double f0_4_1 = p4_1 - p5_1;
  const double f0_5_1 = p4_1 + p5_1;
  const double f0_4_2 = p4_2 - p5_2;
  const double f0_5_2 = p4_2 + p5_2;
  const double f0_6_0 = p6_0 - p7_0;
  const double f0_7_0 = p6_0 + p7_0;
  const double f0_6_1 = p6_1 - p7_1;
  const double f0_7_1 = p6_1 + p7_1;
  const double f0_6_2 = p6_2 - p7_2;
  const double f0_7_2 = p6_2 + p7_2;
  const double f1_0_0 = f0_0_0 - f0_2_0;
  const double f1_2_0 = f0_0_0 + f0_2_0;
  const double f1_0_1 = f0_0_1 - f0_2_1;
  const double f1_2_1 = f0_0_1 + f0_2_1;
  const double f1_0_2 = f0_0_2 - f0_2_2;
  const double f1_2_2 = f0_0_2 + f0_2_2;
  const double f1_1_0 = f0_1_0 - f0_3_0;
  const double f1_3_0 = f0_1_0 + f0_3_0;
  const double f1_1_1 = f0_1_1 - f0_3_1;
  const double f1_3_1 = f0_1_1 + f0_3_1;
  const double f1_1_2 = f0_1_2 - f0_3_2;
  const double f1_3_2 = f0_1_2 + f0_3_2;
  const double f1_4_0 = f0_4_0 - f0_6_0;
  const double f1_6_0 = f0_4_0 + f0_6_0;
  const double f1_4_1 = f0_4_1 - f0_6_1;
  const double f1_6_1 = f0_4_1 + f0_6_1;
  const double f1_4_2 = f0_4_2 - f0_6_2;
  const double f1_6_2 = f0_4_2 + f0_6_2;
  const double f1_5_0 = f0_5_0 - f0_7_0;
  const double f1_7_0 = f0_5_0 + f0_7_0;
  const double f1_5_1 = f0_5_1 - f0_7_1;
  const double f1_7_1 = f0_5_1 + f0_7_1;
  const double f1_5_2 = f0_5_2 - f0_7_2;
  const double f1_7_2 = f0_5_2 + f0_7_2;
  const double f2_0_0 = f1_0_0 - f1_4_0;
  const double f2_4_0 = f1_0_0 + f1_4_0;
  const double f2_0_1 = f1_0_1 - f1_4_1;
  const double f2_4_1 = f1_0_1 + f1_4_1;
  const double f2_0_2 = f1_0_2 - f1_4_2;
  const double f2_4_2 = f1_0_2 + f1_4_2;
  const double f2_1_0 = f1_1_0 - f1_5_0;
  const double f2_5_0 = f1_1_0 + f1_5_0;
  const double f2_1_1 = f1_1_1 - f1_5_1;
  const double f2_5_1 = f1_1_1 + f1_5_1;
  const double f2_1_2 = f1_1_2 - f1_5_2;
  const double f2_5_2 = f1_1_2 + f1_5_2;
  const double f2_2_0 = f1_2_0 - f1_6_0;
  const double f2_6_0 = f1_2_0 + f1_6_0;
  const double f2_2_1 = f1_2_1 - f1_6_1;
  const double f2_6_1 = f1_2_1 + f1_6_1;
  const double f2_2_2 = f1_2_2 - f1_6_2;
  const double f2_6_2 = f1_2_2 + f1_6_2;
  const double f2_3_0 = f1_3_0 - f1_7_0;
  const double f2_7_0 = f1_3_0 + f1_7_0;
  const double f2_3_1 = f1_3_1 - f1_7_1;
  const double f2_7_1 = f1_3_1 + f1_7_1;
  const double f2_3_2 = f1_3_2 - f1_7_2;
  const double f2_7_2 = f1_3_2 + f1_7_2;
  f[0] = f2_0_0;
  f[1] = f2_0_1;
  f[2] = f2_0_2;
  f[3] = f2_1_0;
  f[4] = f2_1_1;
  f[5] = f2_1_2;
  f[6] = f2_2_0;
  f[7] = f2_2_1;
  f[8] = f2_2_2;
  f[9] = f2_3_0;
  f[10] = f2_3_1;
  f[11] = f2_3_2;
  f[12] = f2_4_0;
  f[13] = f2_4_1;
  f[14] = f2_4_2;
  f[15] = f2_5_0;
  f[16] = f2_5_1;
  f[17] = f2_5_2;
  f[18] = f2_6_0;
  f[19] = f2_6_1;
  f[20] = f2_6_2;
  f[21] = f2_7_0;
  f[22] = f2_7_1;
  f[23] = f2_7_2;
}

__device__ __forceinline__ void fi_modal(const double* u, const double* c, double scale, const double* g, double* f) {
  const double q0_0_0 = u[0] + u[3];
  const double q0_1_0 = u[3] - u[0];
  const double q0_0_1 = u[1] + u[4];
  const double q0_1_1 = u[4] - u[1];
  const double q0_0_2 = u[2] + u[5];
  const double q0_1_2 = u[5] - u[2];
  const double q0_2_0 = u[6] + u[9];
  const double q0_3_0 = u[9] - u[6];
  const double q0_2_1 = u[7] + u[10];
  const double q0_3_1 = u[10] - u[7];
  const double q0_2_2 = u[8] + u[11];
  const double q0_3_2 = u[11] - u[8];
  const double q0_4_0 = u[12] + u[15];
  const double q0_5_0 = u[15] - u[12];
  const double q0_4_1 = u[13] + u[16];
  const double q0_5_1 = u[16] - u[13];
  const double q0_4_2 = u[14] + u[17];
  const double q0_5_2 = u[17] - u[14];
  const double q0_6_0 = u[18] + u[21];
  const double q0_7_0 = u[21] - u[18];
  const double q0_6_1 = u[19] + u[22];
  const double q0_7_1 = u[22] - u[19];
  const double q0_6_2 = u[20] + u[23];
  const double q0_7_2 = u[23] - u[20];
  const double q1_0_0 = q0_0_0 + q0_2_0;
  const double q1_2_0 = q0_2_0 - q0_0_0;
  const double q1_0_1 = q0_0_1 + q0_2_1;
  const double q1_2_1 = q0_2_1 - q0_0_1;
  const double q1_0_2 = q0_0_2 + q0_2_2;
  const double q1_2_2 = q0_2_2 - q0_0_2;
  const double q1_1_0 = q0_1_0 + q0_3_0;
  const double q1_3_0 = q0_3_0 - q0_1_0;
  const double q1_1_1 = q0_1_1 + q0_3_1;
  const double q1_3_1 = q0_3_1 - q0_1_1;
  const double q1_1_2 = q0_1_2 + q0_3_2;
  const double q1_3_2 = q0_3_2 - q0_1_2;
  const double q1_4_0 = q0_4_0 + q0_6_0;
  const double q1_6_0 = q0_6_0 - q0_4_0;
  const double q1_4_1 = q0_4_1 + q0_6_1;
  const double q1_6_1 = q0_6_1 - q0_4_1;
  const double q1_4_2 = q0_4_2 + q0_6_2;
  const double q1_6_2 = q0_6_2 - q0_4_2;
  const double q1_5_0 = q0_5_0 + q0_7_0;
  const double q1_7_0 = q0_7_0 - q0_5_0;
  const double q1_5_1 = q0_5_1 + q0_7_1;
  const double q1_7_1 = q0_7_1 - q0_5_1;
  const double q1_5_2 = q0_5_2 + q0_7_2;
  const double q1_7_2 = q0_7_2 - q0_5_2;
  const double q2_0_0 = q1_0_0 + q1_4_0;
  const double q2_4_0 = q1_4_0 - q1_0_0;
  const double q2_0_1 = q1_0_1 + q1_4_1;
  const double q2_4_1 = q1_4_1 - q1_0_1;
  const double q2_0_2 = q1_0_2 + q1_4_2;
  const double q2_4_2 = q1_4_2 - q1_0_2;
  const double q2_1_0 = q1_1_0 + q1_5_0;
  const double q2_5_0 = q1_5_0 - q1_1_0;
  const double q2_1_1 = q1_1_1 + q1_5_1;
  const double q2_5_1 = q1_5_1 - q1_1_1;
  const double q2_1_2 = q1_1_2 + q1_5_2;
  const double q2_5_2 = q1_5_2 - q1_1_2;
  const double q2_2_0 = q1_2_0 + q1_6_0;
  const double q2_6_0 = q1_6_0 - q1_2_0;
  const double q2_2_1 = q1_2_1 + q1_6_1;
  const double q2_6_1 = q1_6_1 - q1_2_1;
  const double q2_2_2 = q1_2_2 + q1_6_2;
  const double q2_6_2 = q1_6_2 - q1_2_2;
  const double q2_3_0 = q1_3_0 + q1_7_0;
  const double q2_7_0 = q1_7_0 - q1_3_0;
  const double q2_3_1 = q1_3_1 + q1_7_1;
  const double q2_7_1 = q1_7_1 - q1_3_1;
  const double q2_3_2 = q1_3_2 + q1_7_2;
  const double q2_7_2 = q1_7_2 - q1_3_2;
  const double e0_0 = g[0] * q2_1_0;
  const double e0_1 = g[1] * q2_2_1;
  const double e0_2 = g[2] * q2_4_2;
  const double e0_3 = g[2] * q2_4_1 + g[1] * q2_2_2;
  const double e0_4 = g[2] * q2_4_0 + g[0] * q2_1_2;
  const double e0_5 = g[1] * q2_2_0 + g[0] * q2_1_1;
  const double s0_0 = (c[0] * e0_0 + c[1] * e0_1 + c[2] * e0_2 + c[3] * e0_3 + c[4] * e0_4 + c[5] * e0_5) * (g[3] / 1.0);
  const double s0_1 = (c[1] * e0_0 + c[6] * e0_1 + c[7] * e0_2 + c[8] * e0_3 + c[9] * e0_4 + c[10] * e0_5) * (g[3] / 1.0);
  const double s0_2 = (c[2] * e0_0 + c[7] * e0_1 + c[11] * e0_2 + c[12] * e0_3 + c[13] * e0_4 + c[14] * e0_5) * (g[3] / 1.0);
  const double s0_3 = (c[3] * e0_0 + c[8] * e0_1 + c[12] * e0_2 + c[15] * e0_3 + c[16] * e0_4 + c[17] * e0_5) * (g[3] / 1.0);
  const double s0_4 = (c[4] * e0_0 + c[9] * e0_1 + c[13] * e0_2 + c[16] * e0_3 + c[18] * e0_4 + c[19] * e0_5) * (g[3] / 1.0);
  const double s0_5 = (c[5] * e0_0 + c[10] * e0_1 + c[14] * e0_2 + c[17] * e0_3 + c[19] * e0_4 + c[20] * e0_5) * (g[3] / 1.0);
  const double e1_1 = g[1] * q2_3_1;
  const double e1_2 = g[2] * q2_5_2;
  const double e1_3 = g[2] * q2_5_1 + g[1] * q2_3_2;
  const double e1_4 = g[2] * q2_5_0;
  const double e1_5 = g[1] * q2_3_0;
  const double s1_1 = (c[6] * e1_1 + c[7] * e1_2 + c[8] * e1_3 + c[9] * e1_4 + c[10] * e1_5) * (g[3] / 3.0);
  const double s1_2 = (c[7] * e1_1 + c[11] * e1_2 + c[12] * e1_3 + c[13] * e1_4 + c[14] * e1_5) * (g[3] / 3.0);
  const double s1_3 = (c[8] * e1_1 + c[12] * e1_2 + c[15] * e1_3 + c[16] * e1_4 + c[17] * e1_5) * (g[3] / 3.0);
  const double s1_4 = (c[9] * e1_1 + c[13] * e1_2 + c[16] * e1_3 + c[18] * e1_4 + c[19] * e1_5) * (g[3] / 3.0);
  const double s1_5 = (c[10] * e1_1 + c[14] * e1_2 + c[17] * e1_3 + c[19] * e1_4 + c[20] * e1_5) * (g[3] / 3.0);
  const double e2_0 = g[0] * q2_3_0;
  const double e2_2 = g[2] * q2_6_2;
  const double e2_3 = g[2] * q2_6_1;
  const double e2_4 = g[2] * q2_6_0 + g[0] * q2_3_2;
  const double e2_5 = g[0] * q2_3_1;
  const double s2_0 = (c[0] * e2_0 + c[2] * e2_2 + c[3] * e2_3 + c[4] * e2_4 + c[5] * e2_5) * (g[3] / 3.0);
  const double s2_2 = (c[2] * e2_0 + c[11] * e2_2 + c[12] * e2_3 + c[13] * e2_4 + c[14] * e2_5) * (g[3] / 3.0);
  const double s2_3 = (c[3] * e2_0 + c[12] * e2_2 + c[15] * e2_3 + c[16] * e2_4 + c[17] * e2_5) * (g[3] / 3.0);
  const double s2_4 = (c[4] * e2_0 + c[13] * e2_2 + c[16] * e2_3 + c[18] * e2_4 + c[19] * e2_5) * (g[3] / 3.0);
  const double s2_5 = (c[5] * e2_0 + c[14] * e2_2 + c[17] * e2_3 + c[19] * e2_4 + c[20] * e2_5) * (g[3] / 3.0);
  const double e4_0 = g[0] * q2_5_0;
  const double e4_1 = g[1] * q2_6_1;
  const double e4_3 = g[1] * q2_6_2;
  const double e4_4 = g[0] * q2_5_2;
  const double e4_5 = g[1] * q2_6_0 + g[0] * q2_5_1;
  const double s4_0 = (c[0] * e4_0 + c[1] * e4_1 + c[3] * e4_3 + c[4] * e4_4 + c[5] * e4_5) * (g[3] / 3.0);
  const double s4_1 = (c[1] * e4_0 + c[6] * e4_1 + c[8] * e4_3 + c[9] * e4_4 + c[10] * e4_5) * (g[3] / 3.0);
  const double s4_3 = (c[3] * e4_0 + c[8] * e4_1 + c[15] * e4_3 + c[16] * e4_4 + c[17] * e4_5) * (g[3] / 3.0);
  const double s4_4 = (c[4] * e4_0 + c[9] * e4_1 + c[16] * e4_3 + c[18] * e4_4 + c[19] * e4_5) * (g[3] / 3.0);
  const double s4_5 = (c[5] * e4_0 + c[10] * e4_1 + c[17] * e4_3 + c[19] * e4_4 + c[20] * e4_5) * (g[3] / 3.0);
  const double e3_2 = g[2] * q2_7_2;
  const double e3_3 = g[2] * q2_7_1;
  const double e3_4 = g[2] * q2_7_0;
  const double s3_2 = (c[11] * e3_2 + c[12] * e3_3 + c[13] * e3_4) * (g[3] / 9.0);
  const double s3_3 = (c[12] * e3_2 + c[15] * e3_3 + c[16] * e3_4) * (g[3] / 9.0);
  const double s3_4 = (c[13] * e3_2 + c[16] * e3_3 + c[18] * e3_4) * (g[3] / 9.0);
  const double e5_1 = g[1] * q2_7_1;
  const double e5_3 = g[1] * q2_7_2;
  const double e5_5 = g[1] * q2_7_0;
  const double s5_1 = (c[6] * e5_1 + c[8] * e5_3 + c[10] * e5_5) * (g[3] / 9.0);
  const double s5_3 = (c[8] * e5_1 + c[15] * e5_3 + c[17] * e5_5) * (g[3] / 9.0);
  const double s5_5 = (c[10] * e5_1 + c[17] * e5_3 + c[20] * e5_5) * (g[3] / 9.0);
  const double e6_0 = g[0] * q2_7_0;
  const double e6_4 = g[0] * q2_7_2;
  const double e6_5 = g[0] * q2_7_1;
  const double s6_0 = (c[0] * e6_0 + c[4] * e6_4 + c[5] * e6_5) * (g[3] / 9.0);
  const double s6_4 = (c[4] * e6_0 + c[18] * e6_4 + c[19] * e6_5) * (g[3] / 9.0);
  const double s6_5 = (c[5] * e6_0 + c[19] * e6_4 + c[20] * e6_5) * (g[3] / 9.0);
  const double p0_0 = 0.0;
  const double p0_1 = 0.0;
  const double p0_2 = 0.0;
  const double p1_0 = g[0] * s0_0;
  const double p1_1 = g[0] * s0_5;
  const double p1_2 = g[0] * s0_4;
  const double p2_0 = g[1] * s0_5;
  const double p2_1 = g[1] * s0_1;
  const double p2_2 = g[1] * s0_3;
  const double p3_0 = g[1] * s1_5 + g[0] * s2_0;
  const double p3_1 = g[1] * s1_1 + g[0] * s2_5;
  const double p3_2 = g[1] * s1_3 + g[0] * s2_4;
  const double p4_0 = g[2] * s0_4;
  const double p4_1 = g[2] * s0_3;
  const double p4_2 = g[2] * s0_2;
  const double p5_0 = g[2] * s1_4 + g[0] * s4_0;
  const double p5_1 = g[2] * s1_3 + g[0] * s4_5;
  const double p5_2 = g[2] * s1_2 + g[0] * s4_4;
  const double p6_0 = g[2] * s2_4 + g[1] * s4_5;
  const double p6_1 = g[2] * s2_3 + g[1] * s4_1;
  const double p6_2 = g[2] * s2_2 + g[1] * s4_3;
  const double p7_0 = g[2] * s3_4 + g[1] * s5_5 + g[0] * s6_0;
  const double p7_1 = g[2] * s3_3 + g[1] * s5_1 + g[0] * s6_5;
  const double p7_2 = g[2] * s3_2 + g[1] * s5_3 + g[0] * s6_4;
  const double f0_0_0 = p0_0 - p1_0;
  const double f0_1_0 = p0_0 + p1_0;
  const double f0_0_1 = p0_1 - p1_1;
  const double f0_1_1 = p0_1 + p1_1;
  const double f0_0_2 = p0_2 - p1_2;
  const double f0_1_2 = p0_2 + p1_2;
  const double f0_2_0 = p2_0 - p3_0;
  const double f0_3_0 = p2_0 + p3_0;
  const double f0_2_1 = p2_1 - p3_1;
  const double f0_3_1 = p2_1 + p3_1;
  const double f0_2_2 = p2_2 - p3_2;
  const double f0_3_2 = p2_2 + p3_2;
  const double f0_4_0 = p4_0 - p5_0;
  const double f0_5_0 = p4_0 + p5_0;
  const double f0_4_1 = p4_1 - p5_1;
  const double f0_5_1 = p4_1 + p5_1;
  const double f0_4_2 = p4_2 - p5_2;
  const double f0_5_2 = p4_2 + p5_2;
  const double f0_6_0 = p6_0 - p7_0;
  const double f0_7_0 = p6_0 + p7_0;
  const double f0_6_1 = p6_1 - p7_1;
  const double f0_7_1 = p6_1 + p7_1;
  const double f0_6_2 = p6_2 - p7_2;
  const double f0_7_2 = p6_2 + p7_2;
  const double f1_0_0 = f0_0_0 - f0_2_0;
  const double f1_2_0 = f0_0_0 + f0_2_0;
  const double f1_0_1 = f0_0_1 - f0_2_1;
  const double f1_2_1 = f0_0_1 + f0_2_1;
  const double f1_0_2 = f0_0_2 - f0_2_2;
  const double f1_2_2 = f0_0_2 + f0_2_2;
  const double f1_1_0 = f0_1_0 - f0_3_0;
  const double f1_3_0 = f0_1_0 + f0_3_0;
  const double f1_1_1 = f0_1_1 - f0_3_1;
  const double f1_3_1 = f0_1_1 + f0_3_1;
  const double f1_1_2 = f0_1_2 - f0_3_2;
  const double f1_3_2 = f0_1_2 + f0_3_2;
  const double f1_4_0 = f0_4_0 - f0_6_0;
  const double f1_6_0 = f0_4_0 + f0_6_0;
  const double f1_4_1 = f0_4_1 - f0_6_1;
  const double f1_6_1 = f0_4_1 + f0_6_1;
  const double f1_4_2 = f0_4_2 - f0_6_2;
  const double f1_6_2 = f0_4_2 + f0_6_2;
  const double f1_5_0 = f0_5_0 - f0_7_0;
  const double f1_7_0 = f0_5_0 + f0_7_0;
  const double f1_5_1 = f0_5_1 - f0_7_1;
  const double f1_7_1 = f0_5_1 + f0_7_1;
  const double f1_5_2 = f0_5_2 - f0_7_2;
  const double f1_7_2 = f0_5_2 + f0_7_2;
  const double f2_0_0 = f1_0_0 - f1_4_0;
  const double f2_4_0 = f1_0_0 + f1_4_0;
  const double f2_0_1 = f1_0_1 - f1_4_1;
  const double f2_4_1 = f1_0_1 + f1_4_1;
  const double f2_0_2 = f1_0_2 - f1_4_2;
  const double f2_4_2 = f1_0_2 + f1_4_2;
  const double f2_1_0 = f1_1_0 - f1_5_0;
  const double f2_5_0 = f1_1_0 + f1_5_0;
  const double f2_1_1 = f1_1_1 - f1_5_1;
  const double f2_5_1 = f1_1_1 + f1_5_1;
  const double f2_1_2 = f1_1_2 - f1_5_2;
  const double f2_5_2 = f1_1_2 + f1_5_2;
  const double f2_2_0 = f1_2_0 - f1_6_0;
  const double f2_6_0 = f1_2_0 + f1_6_0;
  const double f2_2_1 = f1_2_1 - f1_6_1;
  const double f2_6_1 = f1_2_1 + f1_6_1;
  const double f2_2_2 = f1_2_2 - f1_6_2;
  const double f2_6_2 = f1_2_2 + f1_6_2;
  const double f2_3_0 = f1_3_0 - f1_7_0;
  const double f2_7_0 = f1_3_0 + f1_7_0;
  const double f2_3_1 = f1_3_1 - f1_7_1;
  const double f2_7_1 = f1_3_1 + f1_7_1;
  const double f2_3_2 = f1_3_2 - f1_7_2;
  const double f2_7_2 = f1_3_2 + f1_7_2;
  f[0] = f2_0_0;
  f[1] = f2_0_1;
  f[2] = f2_0_2;
  f[3] = f2_1_0;
  f[4] = f2_1_1;
  f[5] = f2_1_2;
  f[6] = f2_2_0;
  f[7] = f2_2_1;
  f[8] = f2_2_2;
  f[9] = f2_3_0;
  f[10] = f2_3_1;
  f[11] = f2_3_2;
  f[12] = f2_4_0;
  f[13] = f2_4_1;
  f[14] = f2_4_2;
  f[15] = f2_5_0;
  f[16] = f2_5_1;
  f[17] = f2_5_2;
  f[18] = f2_6_0;
  f[19] = f2_6_1;
  f[20] = f2_6_2;
  f[21] = f2_7_0;
  f[22] = f2_7_1;
  f[23] = f2_7_2;
}
template<bool fi>
__device__ __forceinline__ void evaluate(const double* u, const double* c, double scale,
                                        const double* g, double* f) {
  if constexpr(fi) fi_modal(u,c,scale,g,f);
  else hg_modal(u,c,scale,g,f);
}

__device__ __forceinline__ int wrap1(int value, int n) {
  return value < 0 ? value+n : (value >= n ? value-n : value);
}

template<bool fi>
__device__ __forceinline__ void element(const double* u,const double* c,const double* scales,
                                       int x,int y,int z,int nx,int ny,int nz,
                                       const double* g,double* f) {
  double local[24];
  #pragma unroll
  for(int a=0;a<8;++a) {
    const int xx=wrap1(x+(a&1),nx), yy=wrap1(y+((a>>1)&1),ny), zz=wrap1(z+((a>>2)&1),nz);
    const int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d) local[3*a+d]=u[3*node+d];
  }
  const int64_t e=(int64_t(z)*ny+y)*nx+x;
  double scale=0.;
  if constexpr(!fi) scale=scales[e];
  evaluate<fi>(local,c+e*21,scale,g,f);
}

template<bool fi,bool atomic>
__global__ void element_kernel(const double* u,const double* c,const double* scales,
    double* output,int nx,int ny,int nz,double gx,double gy,double gz,double volume) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t n=int64_t(nx)*ny*nz;
  if(e>=n)return;
  const int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
  const double g[4]={gx,gy,gz,volume};
  double f[24];
  element<fi>(u,c,scales,x,y,z,nx,ny,nz,g,f);
  #pragma unroll
  for(int a=0;a<8;++a) {
    if constexpr(atomic) {
      const int xx=wrap1(x+(a&1),nx),yy=wrap1(y+((a>>1)&1),ny),zz=wrap1(z+((a>>2)&1),nz);
      const int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
      #pragma unroll
      for(int d=0;d<3;++d) atomicAdd(output+3*node+d,f[3*a+d]);
    } else {
      #pragma unroll
      for(int d=0;d<3;++d) output[24*e+3*a+d]=f[3*a+d];
    }
  }
}

__global__ void gather_kernel(const double* forces,double* output,int nx,int ny,int nz) {
  const int64_t node=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(node>=int64_t(nx)*ny*nz)return;
  const int x=node%nx,y=(node/nx)%ny,z=node/(int64_t(nx)*ny);
  double force[3]={0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;++a) {
    const int ex=wrap1(x-(a&1),nx),ey=wrap1(y-((a>>1)&1),ny),ez=wrap1(z-((a>>2)&1),nz);
    const int64_t e=(int64_t(ez)*ny+ey)*nx+ex;
    #pragma unroll
    for(int d=0;d<3;++d)force[d]+=forces[24*e+3*a+d];
  }
  #pragma unroll
  for(int d=0;d<3;++d)output[3*node+d]=force[d];
}

// Stream z planes through two shared force planes. Unlike a full 3D tile,
// shared storage is O(tx*ty), independent of tz. Every output node has one owner.
template<bool fi>
__global__ void stream_kernel(const double* u,const double* c,const double* scales,
    double* output,int nx,int ny,int nz,int tx,int ty,int tz,
    double gx,double gy,double gz,double volume) {
  extern __shared__ double buffer[];
  const int bx=blockIdx.x%((nx+tx-1)/tx);
  const int by=(blockIdx.x/((nx+tx-1)/tx))%((ny+ty-1)/ty);
  const int bz=blockIdx.x/(((nx+tx-1)/tx)*((ny+ty-1)/ty));
  const int sx=bx*tx,sy=by*ty,sz=bz*tz;
  const int dx=min(tx,nx-sx),dy=min(ty,ny-sy),dz=min(tz,nz-sz);
  const int plane=(dx+1)*(dy+1);
  const double g[4]={gx,gy,gz,volume};
  for(int layer=0;layer<=dz;++layer) {
    double* current=buffer+(layer&1)*plane*24;
    for(int e2=threadIdx.x;e2<plane;e2+=blockDim.x) {
      const int ex=e2%(dx+1),ey=e2/(dx+1);
      const int x=wrap1(sx+ex-1,nx),y=wrap1(sy+ey-1,ny),z=wrap1(sz+layer-1,nz);
      double f[24];
      element<fi>(u,c,scales,x,y,z,nx,ny,nz,g,f);
      #pragma unroll
      for(int j=0;j<24;++j)current[e2*24+j]=f[j];
    }
    __syncthreads();
    if(layer>0) {
      for(int i2=threadIdx.x;i2<dx*dy;i2+=blockDim.x) {
        const int ix=i2%dx,iy=i2/dx;
        double force[3]={0.,0.,0.};
        #pragma unroll
        for(int a=0;a<8;++a) {
          const int ex=ix+1-(a&1),ey=iy+1-((a>>1)&1),ez=layer-((a>>2)&1);
          const double* local=buffer+((ez&1)*plane+ey*(dx+1)+ex)*24+3*a;
          #pragma unroll
          for(int d=0;d<3;++d)force[d]+=local[d];
        }
        const int64_t node=(int64_t(sz+layer-1)*ny+sy+iy)*nx+sx+ix;
        #pragma unroll
        for(int d=0;d<3;++d)output[3*node+d]=force[d];
      }
    }
    __syncthreads();
  }
}

void check(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat64&&t.is_contiguous(),"contiguous CUDA FP64 required");
}

template<bool fi>
void launch(const torch::Tensor& c,const torch::Tensor& scales,const torch::Tensor& u,
    torch::Tensor& output,torch::Tensor& forces,int nx,int ny,int nz,
    double hx,double hy,double hz,int variant,int tx,int ty,int tz,int block) {
  const int64_t n=int64_t(nx)*ny*nz;
  const int blocks=(n+block-1)/block;
  const double gx=1./(4*hx),gy=1./(4*hy),gz=1./(4*hz),volume=hx*hy*hz;
  const auto stream=at::cuda::getCurrentCUDAStream();
  if(variant==0) {
    TORCH_CHECK(forces.numel()==24*n,"two-stage force buffer shape");
    element_kernel<fi,false><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),forces.data_ptr<double>(),nx,ny,nz,gx,gy,gz,volume);
    gather_kernel<<<blocks,block,0,stream>>>(forces.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz);
  } else if(variant==1) {
    output.zero_();
    element_kernel<fi,true><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz,gx,gy,gz,volume);
  } else if(variant==2) {
    const int tiles=((nx+tx-1)/tx)*((ny+ty-1)/ty)*((nz+tz-1)/tz);
    const int shared=2*(tx+1)*(ty+1)*24*sizeof(double);
    if(shared>49152)C10_CUDA_CHECK(cudaFuncSetAttribute(stream_kernel<fi>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    stream_kernel<fi><<<tiles,block,shared,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz,tx,ty,tz,gx,gy,gz,volume);
  } else TORCH_CHECK(false,"unknown variant");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

torch::Tensor apply(torch::Tensor c,torch::Tensor scales,torch::Tensor u,torch::Tensor forces,
    int nz,int ny,int nx,double hx,double hy,double hz,bool fi,int variant,
    int tx,int ty,int tz,int block) {
  check(c);check(scales);check(u);check(forces);
  c10::cuda::CUDAGuard guard(u.device());
  const int64_t n=int64_t(nx)*ny*nz;
  TORCH_CHECK(nx>=2&&ny>=2&&nz>=2&&c.numel()==21*n&&u.numel()==3*n,"B=1 shape required");
  TORCH_CHECK(fi||scales.numel()==n,"HG scale shape");
  TORCH_CHECK(hx>0&&hy>0&&hz>0&&tx>0&&ty>0&&tz>0,"positive geometry/tile required");
  TORCH_CHECK(block==64||block==128||block==256,"block must be 64, 128 or 256");
  auto output=torch::empty_like(u);
  if(fi)launch<true>(c,scales,u,output,forces,nx,ny,nz,hx,hy,hz,variant,tx,ty,tz,block);
  else launch<false>(c,scales,u,output,forces,nx,ny,nz,hx,hy,hz,variant,tx,ty,tz,block);
  return output;
}

pybind11::dict attributes(bool fi,int variant) {
  cudaFuncAttributes a;
  if(fi) {
    if(variant==2)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,stream_kernel<true>));
    else if(variant==1)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true,true>));
    else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true,false>));
  } else {
    if(variant==2)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,stream_kernel<false>));
    else if(variant==1)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<false,true>));
    else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<false,false>));
  }
  pybind11::dict r;r["registers_per_thread"]=a.numRegs;r["local_bytes_per_thread"]=a.localSizeBytes;
  r["static_shared_bytes"]=a.sharedSizeBytes;r["max_threads_per_block"]=a.maxThreadsPerBlock;
  return r;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("attributes",&attributes);}
