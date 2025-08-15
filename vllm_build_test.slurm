#!/bin/bash
#SBATCH --job-name=vllm-build-test
#SBATCH --output=/mnt/polished-lake/home/michaelanderson/vllm-build-%j.out
#SBATCH --error=/mnt/polished-lake/home/michaelanderson/vllm-build-%j.err
#SBATCH --time=01:00:00           # 1 hour should be enough for build + test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # Use multiple CPUs for faster compilation
#SBATCH --mem=32G                 # Compilation can be memory intensive
#SBATCH --gres=gpu:1              # Request 1 GPU for testing
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition name

# Set base paths
BASE_DIR="/mnt/polished-lake/home/michaelanderson"
VLLM_DIR="${BASE_DIR}/vllm-homework"

# Change to vLLM directory
cd "$VLLM_DIR" || exit 1

# Print job information
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Base directory: $BASE_DIR"
echo "========================================="
echo

# Load necessary modules (adjust for your cluster)
# Common module names - uncomment/modify as needed:
# module load cuda/12.1
# module load cudnn/8.9
# module load gcc/11.3
# module load python/3.11

# Check for CUDA
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
else
    echo "Warning: nvidia-smi not found"
fi
echo

# Check for uv in multiple locations
export PATH="$HOME/.local/bin:$BASE_DIR/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH, checking common locations..."
    
    # Check common locations
    for UV_PATH in "$HOME/.local/bin/uv" "$BASE_DIR/.local/bin/uv" "/usr/local/bin/uv"; do
        if [ -x "$UV_PATH" ]; then
            echo "Found uv at: $UV_PATH"
            export PATH="$(dirname "$UV_PATH"):$PATH"
            break
        fi
    done
fi

# Install uv if still not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv to $BASE_DIR/.local/bin..."
    export CARGO_HOME="$BASE_DIR/.cargo"
    export RUSTUP_HOME="$BASE_DIR/.rustup"
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path
    export PATH="$BASE_DIR/.local/bin:$PATH"
    echo "uv installed to: $(which uv)"
else
    echo "Using existing uv: $(which uv)"
fi

# Create virtual environment if it doesn't exist
VENV_DIR="${VLLM_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR with Python 3.11..."
    uv venv "$VENV_DIR" --python 3.11
fi

# Install dependencies
echo "========================================="
echo "Installing dependencies..."
echo "========================================="
uv pip install torch transformers accelerate

# Build vLLM (this is the slow part)
echo "========================================="
echo "Building vLLM from source..."
echo "This may take 15-30 minutes..."
echo "Start time: $(date)"
echo "========================================="

# Set compilation flags for better performance
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Adjust for your GPU architecture

# Install in editable mode
uv pip install -e . --verbose

echo "Build complete at: $(date)"
echo

# Run the test
echo "========================================="
echo "Running inference test..."
echo "========================================="

# Create inline test to avoid file dependency
cat << 'EOF' > /tmp/test_vllm_${SLURM_JOB_ID}.py
import os
import sys
from vllm import LLM, SamplingParams

print(f"Python: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

print("\nInitializing vLLM with facebook/opt-125m...")
llm = LLM(
    model="facebook/opt-125m",
    max_model_len=256,
    gpu_memory_utilization=0.5,
)

sampling_params = SamplingParams(temperature=0.8, max_tokens=30)
prompts = ["Hello world", "The weather is", "Python is"]

print("\nGenerating responses...")
outputs = llm.generate(prompts, sampling_params)

print("\n" + "="*50)
print("Results:")
print("="*50)
for output in outputs:
    print(f"\nPrompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")

print("\n✅ vLLM test successful!")
EOF

uv run python /tmp/test_vllm_${SLURM_JOB_ID}.py

# Cleanup
rm -f /tmp/test_vllm_${SLURM_JOB_ID}.py

echo
echo "========================================="
echo "Job complete at: $(date)"
echo "========================================="