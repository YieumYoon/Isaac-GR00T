#!/bin/bash
# GR00T Fine-tuning 모니터링 스크립트

if [ -z "$1" ]; then
    echo "사용법: ./monitor_training.sh <JOB_ID>"
    echo ""
    echo "현재 실행 중인 Job 찾기:"
    squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
    exit 1
fi

JOB_ID=$1
LOG_DIR="$HOME/Isaac-GR00T/logs"
OUT_LOG="$LOG_DIR/finetune_${JOB_ID}.out"
ERR_LOG="$LOG_DIR/finetune_${JOB_ID}.err"

echo "=========================================="
echo "GR00T Fine-tuning 모니터링 (Job: $JOB_ID)"
echo "=========================================="
echo ""

# Job 상태 확인
echo "📊 Job 상태:"
squeue -j $JOB_ID -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" 2>/dev/null || echo "Job이 대기 중이거나 완료됨"
echo ""

# 실행 중인 노드에서 GPU 사용률 확인
NODE=$(squeue -j $JOB_ID -h -o "%R" 2>/dev/null)
if [ -n "$NODE" ] && [ "$NODE" != "None" ]; then
    echo "🖥️  노드 정보: $NODE"
    echo "GPU 사용률:"
    srun --jobid=$JOB_ID nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || echo "GPU 정보 가져오기 실패"
    echo ""
fi

# 로그 파일 확인
echo "📄 로그 파일:"
if [ -f "$OUT_LOG" ]; then
    echo "✅ Output log: $OUT_LOG"
    echo "   크기: $(ls -lh $OUT_LOG | awk '{print $5}')"
    echo "   마지막 수정: $(stat -c %y $OUT_LOG | cut -d'.' -f1)"
else
    echo "⏳ Output log 아직 생성 안됨"
fi

if [ -f "$ERR_LOG" ]; then
    echo "✅ Error log: $ERR_LOG"
    echo "   크기: $(ls -lh $ERR_LOG | awk '{print $5}')"
    if [ -s "$ERR_LOG" ]; then
        echo "   ⚠️  에러 로그에 내용이 있음!"
    fi
else
    echo "⏳ Error log 아직 생성 안됨"
fi
echo ""

# 학습 진행률 (output 로그에서 추출)
if [ -f "$OUT_LOG" ]; then
    echo "📈 학습 진행률:"
    echo "마지막 10줄:"
    tail -10 "$OUT_LOG"
    echo ""
    
    # Step 정보 찾기
    echo "학습 Step 정보:"
    grep -i "step\|epoch\|loss" "$OUT_LOG" | tail -5 || echo "아직 학습 시작 안됨"
    echo ""
fi

# 에러 확인
if [ -f "$ERR_LOG" ] && [ -s "$ERR_LOG" ]; then
    echo "⚠️  에러 로그 (마지막 20줄):"
    tail -20 "$ERR_LOG"
    echo ""
fi

# 체크포인트 확인
CHECKPOINT_DIR="$HOME/Isaac-GR00T/so101-bimanual-checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "💾 체크포인트 현황:"
    echo "디렉토리: $CHECKPOINT_DIR"
    echo "저장된 파일:"
    ls -lth "$CHECKPOINT_DIR" | head -10 || echo "아직 체크포인트 없음"
else
    echo "💾 체크포인트: 아직 저장 안됨"
fi
echo ""

# 명령어 도움말
echo "=========================================="
echo "유용한 명령어:"
echo "=========================================="
echo "실시간 로그 보기:"
echo "  tail -f $OUT_LOG"
echo ""
echo "에러 로그 확인:"
echo "  tail -f $ERR_LOG"
echo ""
echo "Job 취소:"
echo "  scancel $JOB_ID"
echo ""
echo "노드에 SSH 접속 (Job 실행 중):"
echo "  srun --jobid=$JOB_ID --pty bash"
echo ""
echo "GPU 사용률 실시간 모니터링:"
echo "  watch -n 1 'srun --jobid=$JOB_ID nvidia-smi'"
echo ""

