# SLURM GPU 작업 설정 가이드

## 📋 목차
1. [SLURM 기본 사용법](#slurm-기본-사용법)
2. [그래픽카드별 최적 설정](#그래픽카드별-최적-설정)
3. [작업 제출 및 관리](#작업-제출-및-관리)
4. [모니터링 및 디버깅](#모니터링-및-디버깅)
5. [실전 예시](#실전-예시)

---

## 🚀 SLURM 기본 사용법

### 기본 작업 제출
```bash
# 기본 제출
sbatch 스크립트파일.slurm

# 옵션과 함께 제출
sbatch --job-name=작업이름 --partition=파티션명 스크립트파일.slurm
```

### 주요 SLURM 옵션들
```bash
# 작업 이름
--job-name=gr00t_training

# 파티션 선택
--partition=markov_gpu

# GPU 설정
--gres=gpu:4                    # GPU 4개 사용
--constraint=gpu4090            # RTX 4090 타입 지정
--constraint=gpul40s            # L40S 타입 지정

# 노드 설정
--nodelist=classt25             # 특정 노드 지정
--nodes=1                       # 노드 수
--ntasks=1                      # 태스크 수

# 리소스 설정
--cpus-per-task=32              # CPU 코어 수
--mem=200G                      # 메모리 용량
--time=24:00:00                 # 실행 시간 제한

# 출력 설정
--output=logs/job_%j.out        # 표준 출력 파일
--error=logs/job_%j.err         # 에러 출력 파일
```

---

## 🎯 그래픽카드별 최적 설정

### RTX 4090 (24GB VRAM)
```bash
# 기본 설정
--gres=gpu:4
--constraint=gpu4090
--cpus-per-task=32
--mem=200G
--time=24:00:00

# 학습 설정
--batch-size=24
--gradient-accumulation-steps=1
# 총 effective batch size: 24 × 4 × 1 = 96

# 예상 메모리 사용량: 18-20GB per GPU (75-80% 활용도)
```

### L40S (48GB VRAM)
```bash
# 기본 설정
--gres=gpu:4
--constraint=gpul40s
--cpus-per-task=48
--mem=300G
--time=24:00:00

# 학습 설정
--batch-size=32
--gradient-accumulation-steps=1
# 총 effective batch size: 32 × 4 × 1 = 128

# 예상 메모리 사용량: 25-30GB per GPU (50-60% 활용도)
```

### A100 (80GB VRAM)
```bash
# 기본 설정
--gres=gpu:4
--constraint=gpuA100
--cpus-per-task=64
--mem=500G
--time=24:00:00

# 학습 설정
--batch-size=48
--gradient-accumulation-steps=1
# 총 effective batch size: 48 × 4 × 1 = 192

# 예상 메모리 사용량: 35-40GB per GPU (45-50% 활용도)
```

---

## 📊 리소스 설정 기준

### CPU 설정 기준
| GPU 개수 | 권장 CPU | 이유 |
|----------|----------|------|
| 1개 | 8-16개 | 데이터 로딩, 전처리 |
| 2개 | 16-32개 | GPU 간 통신 오버헤드 |
| 4개 | 32-64개 | NCCL 통신, 병렬 처리 |
| 8개+ | 64-128개 | 대규모 병렬 처리 |

### 메모리 설정 기준
| GPU 타입 | GPU 메모리 | 권장 시스템 메모리 | 비율 |
|----------|------------|-------------------|------|
| RTX 4090 | 24GB | 200-300GB | 4-6배 |
| L40S | 48GB | 300-500GB | 4-6배 |
| A100 | 80GB | 500GB+ | 4-6배 |

---

## 🔧 작업 제출 및 관리

### 작업 제출
```bash
# RTX 4090 노드에 작업 제출
sbatch \
  --job-name=gr00t_rtx4090 \
  --partition=markov_gpu \
  --gres=gpu:4 \
  --constraint=gpu4090 \
  --nodelist=classt25 \
  --cpus-per-task=32 \
  --mem=200G \
  --time=24:00:00 \
  finetune_RTX4090.slurm

# L40S 노드에 작업 제출
sbatch \
  --job-name=gr00t_l40s \
  --partition=markov_gpu \
  --gres=gpu:4 \
  --constraint=gpul40s \
  --nodelist=classt22 \
  --cpus-per-task=48 \
  --mem=300G \
  --time=24:00:00 \
  finetune_L40S.slurm
```

### 작업 상태 확인
```bash
# 내 작업 목록
squeue -u $USER

# 특정 작업 상세 정보
scontrol show job JOBID

# 노드 상태 확인
sinfo -N
sinfo -p markov_gpu
```

### 작업 제어
```bash
# 작업 취소
scancel JOBID

# 작업 일시정지
scontrol suspend JOBID

# 작업 재개
scontrol resume JOBID

# 모든 작업 취소
scancel -u $USER
```

---

## 📈 모니터링 및 디버깅

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/finetune_JOBID.out
tail -f logs/finetune_JOBID.err

# 로그 검색
grep "error\|Error\|ERROR" logs/finetune_JOBID.err
grep "loss\|grad_norm" logs/finetune_JOBID.out
```

### 리소스 사용량 확인
```bash
# GPU 사용량 확인
srun -p markov_gpu -w 노드명 nvidia-smi

# CPU 사용률 확인
srun -p markov_gpu -w 노드명 top -n 1

# 메모리 사용량 확인
srun -p markov_gpu -w 노드명 free -h
```

### 실시간 모니터링
```bash
# 작업 진행 상황 실시간 확인
watch -n 5 'squeue -u $USER'

# 노드 상태 실시간 확인
watch -n 10 'sinfo -N'
```

---

## 🎯 실전 예시

### 현재 사용 가능한 노드 확인
```bash
# GPU 노드 상태 확인
sinfo -N -o "%N %P %T %G" | grep "gpu:" | grep -E "idle|mixed"

# 특정 GPU 타입 노드 확인
sinfo -N -o "%N %P %T %G" | grep "gpu:" | grep -E "idle|mixed" | grep -E "gpu:4|gpu:14"
```

### 환경변수로 설정값 전달
```bash
# 제출 시 환경변수 설정
sbatch --export=BATCH_SIZE=24,ACCUM_STEPS=1,GPU_TYPE=rtx4090 finetune_RTX4090.slurm

# 스크립트에서 환경변수 사용
python scripts/gr00t_finetune.py \
  --batch-size ${BATCH_SIZE:-24} \
  --gradient-accumulation-steps ${ACCUM_STEPS:-1} \
  --gpu-type ${GPU_TYPE:-rtx4090}
```

### 스크립트 템플릿
```bash
#!/bin/bash
#SBATCH --job-name=gr00t_training
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu4090
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# 학습 실행
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/ \
  --num-gpus 4 \
  --output-dir ./checkpoints \
  --max-steps 10000 \
  --batch-size 24 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000
```

---

## ⚠️ 주의사항

### GPU 손실 방지
- **배치 사이즈를 너무 크게 설정하지 말 것**
- **메모리 사용량을 모니터링할 것**
- **온도와 전력 공급을 확인할 것**

### 리소스 최적화
- **CPU가 부족하면**: 데이터 로딩이 느려짐
- **메모리가 부족하면**: OOM 에러 발생
- **리소스가 남으면**: 비용 낭비

### 안전한 설정 권장
- **보수적 설정으로 시작**
- **점진적으로 리소스 증가**
- **모니터링을 통한 최적화**

---

## 📚 유용한 명령어 모음

```bash
# 작업 관련
squeue -u $USER                    # 내 작업 목록
scancel JOBID                      # 작업 취소
scontrol show job JOBID            # 작업 상세 정보

# 노드 관련
sinfo -N                           # 모든 노드 상태
sinfo -p markov_gpu                # GPU 파티션 상태
sinfo -N -o "%N %P %T %G"          # 노드별 GPU 정보

# 모니터링
watch -n 5 'squeue -u $USER'       # 실시간 작업 모니터링
tail -f logs/finetune_JOBID.out    # 실시간 로그 확인
```

---

*이 가이드는 GR00T 모델 파인튜닝을 위한 SLURM 설정을 다룹니다. 실제 사용 시에는 프로젝트 요구사항에 맞게 조정하세요.*
