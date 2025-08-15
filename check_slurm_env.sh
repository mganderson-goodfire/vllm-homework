#!/bin/bash

# Script to check what environment is available in SLURM jobs
echo "========================================="
echo "SLURM Environment Inheritance Check"
echo "========================================="
echo

# First, check current environment
echo "Current Environment (login node):"
echo "---------------------------------"
echo "PATH entries:"
echo "$PATH" | tr ':' '\n' | head -10
echo
echo "Key tools:"
echo "  uv: $(which uv 2>/dev/null || echo 'not found')"
echo "  python3: $(which python3 2>/dev/null || echo 'not found')"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found')"
echo "  module: $(which module 2>/dev/null || echo 'not found')"
echo
echo "Environment variables:"
echo "  HOME: $HOME"
echo "  USER: $USER"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  PYTHONPATH: ${PYTHONPATH:-not set}"
echo

# Create a test SLURM script
cat > test_slurm_env.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=env-test
#SBATCH --output=slurm-env-test-%j.out
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "========================================="
echo "SLURM Job Environment"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo

echo "PATH entries:"
echo "$PATH" | tr ':' '\n' | head -10
echo

echo "Key tools:"
echo "  uv: $(which uv 2>/dev/null || echo 'not found')"
echo "  python3: $(which python3 2>/dev/null || echo 'not found')"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found')"
echo "  module: $(which module 2>/dev/null || echo 'not found')"
echo

echo "Environment variables:"
echo "  HOME: $HOME"
echo "  USER: $USER"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  PYTHONPATH: ${PYTHONPATH:-not set}"
echo "  SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo

echo "Files in ~/.local/bin (if exists):"
if [ -d "$HOME/.local/bin" ]; then
    ls -la "$HOME/.local/bin" | head -10
else
    echo "  Directory does not exist"
fi
echo

echo "Can we install uv?"
if ! command -v uv &> /dev/null; then
    echo "  uv not found, attempting to install..."
    curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path 2>&1 | head -20
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        echo "  ✓ uv installed successfully to $HOME/.local/bin"
        echo "  Version: $(uv --version)"
    else
        echo "  ✗ Failed to install uv"
    fi
else
    echo "  ✓ uv already available"
    echo "  Version: $(uv --version)"
    echo "  Location: $(which uv)"
fi
EOF

echo "Do you want to submit a test job to check SLURM environment? (y/n)"
read -r response

if [[ "$response" == "y" ]]; then
    echo
    echo "Submitting test job..."
    JOB_ID=$(sbatch test_slurm_env.sh | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        echo "Job submitted with ID: $JOB_ID"
        echo
        echo "Waiting for job to complete (max 30 seconds)..."
        
        # Wait for job to complete
        for i in {1..30}; do
            if squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; then
                sleep 1
            else
                break
            fi
        done
        
        OUTPUT_FILE="slurm-env-test-${JOB_ID}.out"
        if [ -f "$OUTPUT_FILE" ]; then
            echo
            echo "Job output:"
            echo "========================================="
            cat "$OUTPUT_FILE"
            echo "========================================="
            echo
            echo "Output saved in: $OUTPUT_FILE"
        else
            echo "Output file not found yet. Check: $OUTPUT_FILE"
        fi
    else
        echo "Failed to submit job"
    fi
    
    # Cleanup
    rm -f test_slurm_env.sh
else
    echo "Test script created as 'test_slurm_env.sh'"
    echo "You can submit it manually with: sbatch test_slurm_env.sh"
fi

echo
echo "========================================="
echo "Best Practices for vLLM SLURM setup:"
echo "========================================="
echo "1. Install uv in your home directory once:"
echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "   This installs to ~/.local/bin/uv"
echo
echo "2. Or install in a shared location:"
echo "   /mnt/polished-lake/home/michaelanderson/.local/bin/uv"
echo
echo "3. The SLURM script checks and installs if needed,"
echo "   but pre-installing saves time on each job"
echo
echo "4. Virtual environments in shared storage persist"
echo "   across jobs: .venv/ in your project directory"