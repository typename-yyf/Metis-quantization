#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>
using namespace std;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // shift address to current block start
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // index for tiling
  const uint threadRow = threadIdx.x / BN;
  const uint threadCol = threadIdx.x % BN;

  // init SMEM;
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // index for SMEM load;
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  // allocate thread-local results in registerfile
  float threadResults[TM] = {0.0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK){
    As[threadIdx.x] = A[innerRowA * K + innerColA];
    Bs[threadIdx.x] = B[innerRowB * N + innerColB];
    __syncthreads();
    
    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; dotIdx++){
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; resIdx++){
        threadResults[resIdx] += 
          As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; resIdx++){
    C[(threadRow * TM + resIdx) * N + threadCol] = 
      alpha * threadResults[resIdx] + 
      beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}


void launch_matmul_my_4(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float alpha = 1.0;
    float beta = 0.0;

    runSgemm1DBlocktiling(M, N, K, alpha, A.data_ptr<float>(), B.data_ptr<float>(), beta, C.data_ptr<float>());    
}