#!/bin/bash

echo "========================================="
echo "GPU Configuration Check"
echo "========================================="

# Basic GPU info
echo "1. GPU Model and Driver:"
nvidia-smi --query-gpu=name,driver_version --format=csv
echo

echo "2. CUDA Version:"
nvidia-smi | grep "CUDA Version"
echo

echo "3. Compute Capability:"
nvidia-smi --query-gpu=compute_cap --format=csv
echo

echo "4. GPU Memory:"
nvidia-smi --query-gpu=memory.total --format=csv
echo

echo "5. Full nvidia-smi output:"
nvidia-smi
echo

# Check for CUDA toolkit
echo "6. CUDA Toolkit (nvcc):"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH"
    echo "Checking common locations..."
    for path in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        if [ -x "$path" ]; then
            echo "Found at: $path"
            $path --version | grep release
        fi
    done
fi
echo

# Generate TORCH_CUDA_ARCH_LIST recommendation
echo "========================================="
echo "Recommendations for vllm_build_test.slurm:"
echo "========================================="
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "Set TORCH_CUDA_ARCH_LIST=\"${COMPUTE_CAP}\""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU detected: $GPU_NAME"

if echo "$GPU_NAME" | grep -q "H100"; then
    echo "→ H100 (Hopper): Use TORCH_CUDA_ARCH_LIST=\"9.0\""
elif echo "$GPU_NAME" | grep -q "A100"; then
    echo "→ A100 (Ampere): Use TORCH_CUDA_ARCH_LIST=\"8.0\""
elif echo "$GPU_NAME" | grep -q "V100"; then
    echo "→ V100 (Volta): Use TORCH_CUDA_ARCH_LIST=\"7.0\""
fi