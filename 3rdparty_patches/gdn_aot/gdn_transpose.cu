// Per-head in-place transpose of the GDN recurrent state's last two dims.
// FlashInfer writes h_state as S[v][k] ([nv][N][N], N=head_dim); Atlas's decode
// kernel reads H[k*v_dim+v] = S[k][v]. Swap (i,j)<->(j,i) per head (diagonal fixed).
#include <cuda_runtime.h>
__global__ void k_transpose_heads_sq(float* S, int N) {
    int head = blockIdx.y;
    int lin = blockIdx.x * blockDim.x + threadIdx.x; // 0..N*N-1
    int i = lin / N, j = lin % N;
    if (lin < N * N && j < i) {                      // lower<->upper, once each
        float* H = S + (size_t)head * N * N;
        float a = H[i * N + j];
        H[i * N + j] = H[j * N + i];
        H[j * N + i] = a;
    }
}
extern "C" void atlas_transpose_heads(float* S, int nheads, int N, void* stream) {
    dim3 block(256);
    dim3 grid((N * N + 255) / 256, nheads);
    k_transpose_heads_sq<<<grid, block, 0, (cudaStream_t)stream>>>(S, N);
}
