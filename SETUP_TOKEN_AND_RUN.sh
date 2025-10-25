#!/bin/bash
# Hugging Face 토큰 설정 및 Job 재시작 스크립트

echo "================================================"
echo "Hugging Face 토큰 설정 및 학습 재시작"
echo "================================================"
echo ""

# 1. 토큰 확인
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "✅ Hugging Face 토큰이 이미 설정되어 있습니다!"
    TOKEN_PREVIEW=$(head -c 10 $HOME/.cache/huggingface/token)
    echo "   토큰 시작: ${TOKEN_PREVIEW}..."
else
    echo "❌ Hugging Face 토큰이 없습니다!"
    echo ""
    echo "다음 방법 중 하나로 설정하세요:"
    echo ""
    echo "방법 1: 대화형"
    echo "  huggingface-cli login"
    echo ""
    echo "방법 2: 직접 입력"
    echo "  mkdir -p ~/.cache/huggingface"
    echo "  echo 'hf_YOUR_TOKEN' > ~/.cache/huggingface/token"
    echo "  chmod 600 ~/.cache/huggingface/token"
    echo ""
    echo "토큰 받기: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

echo ""
echo "2. 이전 Job 확인..."
RUNNING_JOBS=$(squeue -u $USER -h -o "%i" | wc -l)
if [ $RUNNING_JOBS -gt 0 ]; then
    echo "⚠️  실행 중인 Job이 있습니다:"
    squeue -u $USER
    echo ""
    read -p "모두 취소하고 재시작하시겠습니까? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        squeue -u $USER -h -o "%i" | xargs scancel
        echo "✅ Job 취소 완료"
    else
        echo "취소되었습니다."
        exit 0
    fi
fi

echo ""
echo "3. 새 Job 제출 중..."
cd ~/Isaac-GR00T
sbatch finetune_L40S.slurm

echo ""
echo "4. Job 상태 확인:"
sleep 2
squeue -u $USER

echo ""
echo "================================================"
echo "완료! 로그 확인:"
echo "  tail -f ~/Isaac-GR00T/logs/finetune_*.out"
echo "================================================"

