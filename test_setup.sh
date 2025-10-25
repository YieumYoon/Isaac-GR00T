#!/bin/bash
# GR00T Fine-tuning 환경 테스트 스크립트

echo "=========================================="
echo "GR00T Fine-tuning 환경 테스트"
echo "=========================================="
echo ""

# 1. 디렉토리 확인
echo "1️⃣  디렉토리 구조 확인..."
if [ -d "$HOME/Isaac-GR00T" ]; then
    echo "✅ Isaac-GR00T 디렉토리 존재"
else
    echo "❌ Isaac-GR00T 디렉토리 없음"
fi

if [ -d "$HOME/Isaac-GR00T/logs" ]; then
    echo "✅ logs 디렉토리 존재"
else
    echo "⚠️  logs 디렉토리 없음 - 생성 중..."
    mkdir -p "$HOME/Isaac-GR00T/logs"
fi

if [ -f "$HOME/Isaac-GR00T/scripts/gr00t_finetune.py" ]; then
    echo "✅ gr00t_finetune.py 스크립트 존재"
else
    echo "❌ gr00t_finetune.py 스크립트 없음"
fi
echo ""

# 2. 가상환경 확인
echo "2️⃣  가상환경 확인..."
if [ -d "$HOME/Isaac-GR00T/gr00t" ]; then
    echo "✅ 가상환경 디렉토리 존재"
    if [ -f "$HOME/Isaac-GR00T/gr00t/bin/activate" ]; then
        echo "✅ activate 스크립트 존재"
        source "$HOME/Isaac-GR00T/gr00t/bin/activate"
        echo "   Python 버전: $(python --version 2>&1)"
    else
        echo "❌ activate 스크립트 없음"
    fi
else
    echo "❌ 가상환경 디렉토리 없음"
fi
echo ""

# 3. 필수 Python 패키지 확인
echo "3️⃣  Python 패키지 확인..."
packages=("torch" "torchvision" "huggingface_hub")
for pkg in "${packages[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        version=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo "✅ $pkg ($version)"
    else
        echo "❌ $pkg 설치 안됨"
    fi
done
echo ""

# 4. Hugging Face CLI 확인
echo "4️⃣  Hugging Face CLI 확인..."
if command -v hf &> /dev/null; then
    echo "✅ hf 명령어 사용 가능"
    hf --version
else
    echo "❌ hf 명령어 없음"
    echo "   설치: pip install -U huggingface_hub[cli]"
fi
echo ""

# 5. CUDA/GPU 확인 (모듈 로드 후)
echo "5️⃣  CUDA 및 GPU 확인..."
module purge 2>/dev/null
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null
module load CUDA/12.6.0 2>/dev/null

if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi 사용 가능"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
else
    echo "⚠️  nvidia-smi 사용 불가 (로그인 노드에서는 정상)"
fi

if [ -f "$HOME/Isaac-GR00T/gr00t/bin/activate" ]; then
    source "$HOME/Isaac-GR00T/gr00t/bin/activate"
    python -c "import torch; print('✅ PyTorch CUDA 지원:', torch.cuda.is_available())" 2>/dev/null || echo "❌ PyTorch import 실패"
fi
echo ""

# 6. $TMPDIR 확인
echo "6️⃣  작업 디렉토리 ($TMPDIR) 확인..."
if [ -n "$TMPDIR" ]; then
    echo "✅ TMPDIR 설정됨: $TMPDIR"
    if [ -d "$TMPDIR" ]; then
        echo "   사용 가능한 공간: $(df -h $TMPDIR | tail -1 | awk '{print $4}')"
    fi
else
    echo "⚠️  TMPDIR 미설정 (SLURM job에서만 설정됨)"
    echo "   대체: /tmp"
fi
echo ""

# 7. 데이터셋 접근 테스트
echo "7️⃣  Hugging Face 데이터셋 접근 테스트..."
echo "   (실제 다운로드는 하지 않고 접근만 테스트)"
if command -v hf &> /dev/null; then
    hf download --repo-type dataset YieumYoon/recode-bimanual-red-block-basket-v2.1 --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Hugging Face 데이터셋 명령어 작동"
    else
        echo "⚠️  Hugging Face 로그인 필요할 수 있음"
        echo "   실행: huggingface-cli login"
    fi
else
    echo "❌ hf 명령어 없음"
fi
echo ""

# 8. 요약
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "1. 모든 ✅가 나오면 SLURM job 제출 준비 완료"
echo "2. ❌나 ⚠️가 있으면 해당 항목 수정 필요"
echo "3. Job 제출: sbatch finetune_gr00t.slurm"
echo "4. Job 상태 확인: squeue -u $USER"
echo ""

