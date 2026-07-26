// SPDX-License-Identifier: AGPL-3.0-only
//
// cuBLASLt stub for the HIP target. spark links `-lcublasLt` unconditionally,
// but the cuBLASLt GEMM path is opt-in (ATLAS_CUBLAS_GEMM=1) — the default
// runtime uses Atlas's own hand-written GEMMs. There is no hipBLASLt mapping
// here yet, so every entry point returns a non-success cuBLASLt status (1 =
// CUBLAS_STATUS_NOT_INITIALIZED). If ATLAS_CUBLAS_GEMM is ever set on AMD, the
// create call fails and the caller must fall back rather than run a wrong GEMM.
// Porting these to hipBLASLt (hipblasLtCreate/hipblasLtMatmul/…) is future work.
//
// extern "C", matched by name at link time — arguments are irrelevant to
// resolution, so the stubs are parameterless.
extern "C" {

int cublasLtCreate() { return 1; }
int cublasLtDestroy() { return 1; }
int cublasLtMatmul() { return 1; }
int cublasLtMatmulAlgoGetHeuristic() { return 1; }
int cublasLtMatmulDescCreate() { return 1; }
int cublasLtMatmulDescDestroy() { return 1; }
int cublasLtMatmulDescSetAttribute() { return 1; }
int cublasLtMatmulPreferenceCreate() { return 1; }
int cublasLtMatmulPreferenceDestroy() { return 1; }
int cublasLtMatmulPreferenceSetAttribute() { return 1; }
int cublasLtMatrixLayoutCreate() { return 1; }
int cublasLtMatrixLayoutDestroy() { return 1; }

}  // extern "C"
