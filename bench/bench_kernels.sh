#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Running Atlas Kernel Microbenchmarks (Serially)"
echo "=========================================================="

# Ensure CUDA is in PATH for nvcc matching build.rs
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# Run each kernel microbenchmark sequentially to avoid GPU contention
# which would skew Criterion latency measurements.

echo ""
echo "=> Benchmarking: Normalization (RMSNorm, Gated RMSNorm)"
cargo bench -p atlas-spark-bench --bench norm

echo ""
echo "=> Benchmarking: Activations (SiLUxMul)"
cargo bench -p atlas-spark-bench --bench activation

echo ""
echo "=> Benchmarking: GEMM (Dense TC, W4A16)"
cargo bench -p atlas-spark-bench --bench gemm

echo ""
echo "=> Benchmarking: RoPE (Rotary Position Embeddings)"
cargo bench -p atlas-spark-bench --bench rope

echo ""
echo "=> Benchmarking: MoE (Grouped W4A16 GEMM)"
cargo bench -p atlas-spark-bench --bench moe

echo ""
echo "=> Benchmarking: Attention (Paged Decode FP8)"
cargo bench -p atlas-spark-bench --bench attention

echo ""
echo "=> Benchmarking: SSM (Causal Conv1d, Gated Delta Rule)"
cargo bench -p atlas-spark-bench --bench ssm

echo ""
echo "=========================================================="
echo " All kernel microbenchmarks completed successfully!"
echo "=========================================================="
