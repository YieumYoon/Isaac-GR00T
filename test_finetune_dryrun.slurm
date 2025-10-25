#!/bin/bash
#SBATCH --job-name=gr00t_test
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1              # 테스트는 1개 GPU만 사용
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00           # 30분만 테스트
#SBATCH --output=/home/jxl2244/Isaac-GR00T/logs/test_%j.out
#SBATCH --error=/home/jxl2244/Isaac-GR00T/logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Error handling
set -euo pipefail

echo "=========================================="
echo "GR00T Fine-tuning 테스트 (Dry Run)"
echo "=========================================="
echo ""

# 1. SLURM 환경 정보
echo "1️⃣  SLURM 환경 정보"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# 2. 작업 디렉토리 테스트
echo "2️⃣  작업 디렉토리 ($TMPDIR) 테스트"
WORK_DIR="$TMPDIR"
echo "TMPDIR: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo "사용 가능한 공간: $(df -h . | tail -1 | awk '{print $4}')"
echo "현재 디렉토리: $(pwd)"
echo ""

# 3. 프로젝트 복사 테스트
echo "3️⃣  프로젝트 복사 테스트"
echo "복사 시작 시간: $(date)"
time cp -r $HOME/Isaac-GR00T .
cd Isaac-GR00T
echo "복사된 파일 크기: $(du -sh . | cut -f1)"
echo "스크립트 존재 확인:"
ls -lh scripts/gr00t_finetune.py 2>/dev/null && echo "✅ gr00t_finetune.py 존재" || echo "❌ 스크립트 없음"
echo ""

# 4. 모듈 로드 테스트
echo "4️⃣  모듈 로드 테스트"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.6.0
echo "✅ 모듈 로드 완료"
echo ""

# 5. 가상환경 테스트
echo "5️⃣  가상환경 활성화 테스트"
source $HOME/Isaac-GR00T/gr00t/bin/activate
echo "Python: $(python --version)"
echo "Python 경로: $(which python)"
echo ""

# 6. PyTorch 및 CUDA 테스트
echo "6️⃣  PyTorch 및 CUDA 테스트"
python << 'EOF'
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"GPU 개수: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  메모리: {mem:.1f} GB")
EOF
echo ""

# 7. Hugging Face CLI 테스트
echo "7️⃣  Hugging Face CLI 테스트"
if command -v hf &> /dev/null; then
    echo "✅ hf 명령어 사용 가능"
    hf --version
    echo ""
    echo "샘플 데이터셋 정보 가져오기 (다운로드 안함):"
    hf ls --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-v2.1 | head -5 || echo "⚠️  데이터셋 접근 실패 (로그인 필요할 수 있음)"
else
    echo "❌ hf 명령어 없음"
    echo "pip install -U huggingface_hub[cli] 필요"
fi
echo ""

# 8. 작은 데이터셋 다운로드 테스트 (선택적)
echo "8️⃣  데이터셋 다운로드 테스트 (스킵 - 시간 절약)"
echo "실제 SLURM job에서 다운로드 됨"
mkdir -p demo_data
echo ""

# 9. 환경 변수 테스트
echo "9️⃣  GPU 환경 변수 테스트"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# 10. 스크립트 실행 가능성 테스트 (help만)
echo "🔟 Fine-tuning 스크립트 테스트"
if [ -f "scripts/gr00t_finetune.py" ]; then
    echo "스크립트 help 출력:"
    python scripts/gr00t_finetune.py --help || echo "⚠️  스크립트 실행 중 오류"
else
    echo "❌ scripts/gr00t_finetune.py 없음"
fi
echo ""

# 11. 체크포인트 디렉토리 테스트
echo "1️⃣1️⃣  출력 디렉토리 테스트"
mkdir -p checkpoints
mkdir -p logs
echo "✅ checkpoints 디렉토리 생성됨: $(pwd)/checkpoints"
echo "✅ logs 디렉토리 생성됨: $(pwd)/logs"
echo ""

# 12. rsync 테스트
echo "1️⃣2️⃣  결과 복사 테스트"
echo "테스트 파일 생성..."
echo "test checkpoint" > checkpoints/test.txt
mkdir -p $HOME/Isaac-GR00T/test-checkpoints
rsync -av --progress checkpoints/test.txt $HOME/Isaac-GR00T/test-checkpoints/
if [ -f "$HOME/Isaac-GR00T/test-checkpoints/test.txt" ]; then
    echo "✅ rsync 작동 확인"
    rm -rf $HOME/Isaac-GR00T/test-checkpoints
else
    echo "❌ rsync 실패"
fi
echo ""

# 최종 요약
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="
echo ""
echo "✅ 모든 테스트가 성공하면 실제 fine-tuning 실행 가능"
echo ""
echo "실제 fine-tuning 시작:"
echo "  sbatch finetune_gr00t.slurm"
echo ""
echo "Job 모니터링:"
echo "  squeue -u $USER                                    # Job 상태"
echo "  tail -f ~/Isaac-GR00T/logs/finetune_<JOB_ID>.out  # 실시간 로그"
echo "  tail -f ~/Isaac-GR00T/logs/finetune_<JOB_ID>.err  # 에러 로그"
echo ""

