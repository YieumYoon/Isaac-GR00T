#!/bin/bash
# 체크포인트 자동 업로드 & 정리 스크립트
# 백그라운드에서 실행되어 새 체크포인트를 감지하고 업로드

CHECKPOINT_DIR="$1"
HF_REPO="$2"
CHECK_INTERVAL=300  # 5분마다 확인

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$HF_REPO" ]; then
    echo "Usage: $0 <checkpoint_dir> <hf_repo>"
    exit 1
fi

echo "========================================"
echo "체크포인트 자동 업로더 시작"
echo "감시 디렉토리: $CHECKPOINT_DIR"
echo "업로드 대상: $HF_REPO"
echo "========================================"

# Hugging Face repo 생성 (없으면)
huggingface-cli repo create "$HF_REPO" --type model --private || true

uploaded_checkpoints=()

while true; do
    # 새 체크포인트 찾기
    for checkpoint in $(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort); do
        checkpoint_name=$(basename "$checkpoint")
        
        # 이미 업로드했는지 확인
        if [[ " ${uploaded_checkpoints[@]} " =~ " ${checkpoint_name} " ]]; then
            continue
        fi
        
        # 체크포인트가 완전히 저장되었는지 확인 (5분 이상 수정 안됨)
        last_modified=$(find "$checkpoint" -type f -printf '%T@\n' | sort -n | tail -1)
        current_time=$(date +%s)
        age=$((current_time - ${last_modified%.*}))
        
        if [ $age -lt 300 ]; then
            echo "[$checkpoint_name] 아직 저장 중... (${age}초 전)"
            continue
        fi
        
        echo ""
        echo "========================================"
        echo "[$checkpoint_name] 업로드 시작..."
        echo "========================================"
        
        # Hugging Face Hub에 업로드
        if huggingface-cli upload "$HF_REPO" "$checkpoint" "$checkpoint_name" --private; then
            echo "✅ 업로드 성공: $checkpoint_name"
            uploaded_checkpoints+=("$checkpoint_name")
            
            # 업로드 성공 시 로컬 파일 삭제 (최신 2개 제외)
            all_checkpoints=($(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort))
            num_checkpoints=${#all_checkpoints[@]}
            
            if [ $num_checkpoints -gt 2 ]; then
                # 가장 오래된 체크포인트 삭제
                oldest="${all_checkpoints[0]}"
                echo "🗑️  로컬 삭제: $(basename $oldest) (최신 2개 유지)"
                rm -rf "$oldest"
            fi
        else
            echo "❌ 업로드 실패: $checkpoint_name"
        fi
    done
    
    # 다음 확인까지 대기
    sleep $CHECK_INTERVAL
done

