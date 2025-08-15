#!/bin/bash

# Helper script to submit vLLM build/test job to SLURM

echo "vLLM SLURM Job Submission"
echo "========================="

# Check if we're on a SLURM cluster
if ! command -v sbatch &> /dev/null; then
    echo "Error: sbatch not found. Are you on a SLURM cluster?"
    exit 1
fi

# Parse optional arguments
PARTITION=""
TIME="01:00:00"
GPUS="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --partition NAME   SLURM partition (default: auto-detect)"
            echo "  -t, --time TIME        Time limit (default: 01:00:00)"
            echo "  -g, --gpus NUM         Number of GPUs (default: 1)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If partition not specified, try to detect GPU partition
if [ -z "$PARTITION" ]; then
    echo "Detecting available GPU partitions..."
    GPU_PARTITIONS=$(sinfo -o "%P" | grep -E "gpu|GPU|tesla|a100|v100" | head -1 | tr -d '*')
    if [ -n "$GPU_PARTITIONS" ]; then
        PARTITION=$(echo $GPU_PARTITIONS | cut -d' ' -f1)
        echo "Found GPU partition: $PARTITION"
    else
        echo "Warning: Could not auto-detect GPU partition"
        echo "Available partitions:"
        sinfo -o "%20P %5a %10l %10L %10s %10z"
        echo
        read -p "Enter partition name: " PARTITION
    fi
fi

# Set base paths
BASE_DIR="/mnt/polished-lake/home/michaelanderson"
VLLM_DIR="${BASE_DIR}/vllm-homework"

# Create a custom SLURM script with the right parameters
CUSTOM_SCRIPT="${BASE_DIR}/vllm_build_test_custom.slurm"
cat > "$CUSTOM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=vllm-build-test
#SBATCH --output=${BASE_DIR}/vllm-build-%j.out
#SBATCH --error=${BASE_DIR}/vllm-build-%j.err
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:$GPUS
#SBATCH --partition=$PARTITION

$(cat vllm_build_test.slurm | tail -n +12)  # Skip the SBATCH headers
EOF

# Submit the job
echo
echo "Submitting job with:"
echo "  Partition: $PARTITION"
echo "  Time limit: $TIME"
echo "  GPUs: $GPUS"
echo "  Output dir: $BASE_DIR"
echo

JOB_ID=$(sbatch "$CUSTOM_SCRIPT" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo
    echo "Monitor your job with:"
    echo "  squeue -j $JOB_ID                              # Check job status"
    echo "  tail -f ${BASE_DIR}/vllm-build-$JOB_ID.out    # Watch output"
    echo "  tail -f ${BASE_DIR}/vllm-build-$JOB_ID.err    # Watch errors"
    echo "  scancel $JOB_ID                                # Cancel if needed"
else
    echo "❌ Failed to submit job"
    exit 1
fi

# Note: Keeping the custom script for debugging if needed
echo
echo "Custom SLURM script saved at: $CUSTOM_SCRIPT"