#!/bin/bash
#SBATCH --job-name=gr00t_rtx4090
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:4               # 4 GPUs
#SBATCH --constraint=gpu4090       # RTX 4090 타입 지정
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jxl2244/Isaac-GR00T/logs/finetune_%j.out
#SBATCH --error=/home/jxl2244/Isaac-GR00T/logs/finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Error handling
set -euo pipefail

# Use node-local scratch space
WORK_DIR="$TMPDIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $WORK_DIR"
echo "Available space: $(df -h . | tail -1 | awk '{print $4}')"
echo "========================================"

# Load modules FIRST
echo "Loading modules..."
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.6.0

# Create venv in scratch
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install huggingface_hub CLI FIRST (required for data download)
echo "Installing huggingface_hub CLI..."
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir "huggingface_hub[cli]"
echo "huggingface_hub version: $(pip show huggingface_hub | grep Version)"

# Copy project code
echo "Copying project to scratch..."
cp -r $HOME/Isaac-GR00T .
cd Isaac-GR00T

# Setup Hugging Face token (if exists)
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "Found Hugging Face token, setting up authentication..."
    export HF_TOKEN=$(cat $HOME/.cache/huggingface/token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
else
    echo "WARNING: No Hugging Face token found. May hit rate limits!"
fi

# Download datasets
echo "Downloading datasets..."
mkdir -p demo_data

echo "Dataset 1/3: recode-bimanual-red-block-basket-v2.1..."
hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-v2.1

echo "Dataset 2/3: recode-bimanual-red-block-basket-2-v2.1..."
hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-2-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-2-v2.1

echo "Dataset 3/3: recode-bimanual-red-block-basket-4-v2.1..."
hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-4-v2.1 \
    --local-dir ./demo_data/recode-bimanual-red-block-basket-4-v2.1

echo "Datasets downloaded!"

# Copy modality.json to each dataset (CRITICAL for proper data loading!)
echo "Copying modality.json to datasets..."
for dataset in recode-bimanual-red-block-basket-v2.1 recode-bimanual-red-block-basket-2-v2.1 recode-bimanual-red-block-basket-4-v2.1; do
    mkdir -p ./demo_data/$dataset/meta/
    # Use the modality.json from home directory
    if [ -f "$HOME/demo_data/recode-bimanual-red-block-basket-2-v2.1/meta/modality.json" ]; then
        cp $HOME/demo_data/recode-bimanual-red-block-basket-2-v2.1/meta/modality.json ./demo_data/$dataset/meta/
        echo "  ✓ Copied modality.json to $dataset"
    else
        echo "  ⚠ WARNING: modality.json not found in home!"
    fi
done

# Fix stats.json count field (expand from 1D to 12D)
echo "Fixing stats.json count fields..."
for dataset in recode-bimanual-red-block-basket-v2.1 recode-bimanual-red-block-basket-2-v2.1 recode-bimanual-red-block-basket-4-v2.1; do
    python3 << EOF
import json
stats_path = './demo_data/$dataset/meta/stats.json'
with open(stats_path, 'r') as f:
    stats = json.load(f)
# Fix observation.state count
if 'observation.state' in stats and 'count' in stats['observation.state']:
    count_val = stats['observation.state']['count'][0]
    stats['observation.state']['count'] = [count_val] * 12
# Fix action count
if 'action' in stats and 'count' in stats['action']:
    count_val = stats['action']['count'][0]
    stats['action']['count'] = [count_val] * 12
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  ✓ Fixed stats.json for $dataset")
EOF
done

# Copy data config to project root (needed for import)
echo "Copying recode_data_config.py..."
cp $HOME/Isaac-GR00T/recode_data_config.py .

# Install gr00t package
echo "Installing gr00t package..."
pip install --no-cache-dir -e .[base]

# Install flash-attn (for 2-4x speedup!)
echo "Installing flash-attention..."
pip install --no-build-isolation --no-cache-dir flash-attn==2.7.1.post4 || echo "⚠️  Flash-attn install failed, continuing without it"
echo "Installation complete!"

# Verify environment
echo "========================================"
echo "Environment Info:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in {0..3}; do
    python -c "import torch; print(f'GPU $i: {torch.cuda.get_device_name($i)}')" 2>/dev/null || true
done
echo "========================================"

# Create output directories
mkdir -p checkpoints logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# Setup automatic checkpoint uploader
HF_REPO="$USER/gr00t-so101-bimanual-$(date +%Y%m%d-%H%M)"
echo "Checkpoints will be auto-uploaded to: $HF_REPO"
echo "This saves home directory space (only 10GB free)"

# Copy uploader script and start in background
cp $HOME/Isaac-GR00T/checkpoint_uploader.sh .
chmod +x checkpoint_uploader.sh
./checkpoint_uploader.sh ./checkpoints "$HF_REPO" > checkpoint_uploader.log 2>&1 &
UPLOADER_PID=$!
echo "Auto-uploader started (PID: $UPLOADER_PID)"

# Run fine-tuning (RTX 4090 optimized: 24GB VRAM per GPU)
echo "Starting fine-tuning..."
echo "Batch size: 24 × 4 GPUs × 1 accum = 96 effective (balanced for RTX 4090)"
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/recode-bimanual-red-block-basket-v2.1/ \
                 ./demo_data/recode-bimanual-red-block-basket-2-v2.1/ \
                 ./demo_data/recode-bimanual-red-block-basket-4-v2.1/ \
  --num-gpus 4 \
  --output-dir ./checkpoints \
  --max-steps 10000 \
  --batch-size 24 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000 \
  --data-config recode_data_config:RecodeBimanualDataConfig \
  --video-backend torchvision_av \
  --embodiment-tag new_embodiment

# Stop uploader
kill $UPLOADER_PID 2>/dev/null || true
echo "Auto-uploader stopped"

# Copy results back
echo "Copying results to home..."
mkdir -p $HOME/Isaac-GR00T/so101-bimanual-checkpoints
rsync -av --progress checkpoints/ $HOME/Isaac-GR00T/so101-bimanual-checkpoints/

echo "Training completed! Time: $SECONDS seconds"
echo "Checkpoints: $HOME/Isaac-GR00T/so101-bimanual-checkpoints/"

