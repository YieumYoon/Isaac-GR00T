# GR00T Fine-tuning 체크리스트 및 가이드

## 🎯 스크립트 작동 원리

### 전체 워크플로우
```
1. SLURM이 4-GPU 노드 할당
2. 노드의 빠른 scratch 공간($TMPDIR)으로 프로젝트 복사
3. Hugging Face에서 데이터셋 다운로드 (scratch에 직접)
4. Python 환경 및 CUDA 로드
5. 4-GPU 분산 학습으로 fine-tuning 실행
6. 완료 후 체크포인트를 home 디렉토리로 복사
```

### 주요 장점
- ✅ **빠른 I/O**: scratch 공간 사용으로 데이터 로딩 속도 향상
- ✅ **공간 절약**: home 디렉토리 할당량 절약
- ✅ **자동 정리**: job 종료 시 scratch 자동 삭제
- ✅ **멀티 GPU**: 4-GPU 병렬 처리로 학습 속도 4배

---

## ✅ 사전 확인 체크리스트

### 1단계: 기본 환경 테스트
```bash
cd ~/Isaac-GR00T
./test_setup.sh
```

**확인 항목:**
- [ ] Isaac-GR00T 디렉토리 존재
- [ ] `logs/` 디렉토리 존재 (자동 생성됨)
- [ ] `scripts/gr00t_finetune.py` 파일 존재
- [ ] 가상환경 `gr00t/` 존재 및 활성화 가능
- [ ] PyTorch, torchvision, huggingface_hub 설치됨
- [ ] `hf` 명령어 사용 가능

### 2단계: SLURM 환경 테스트 (30분 소요)
```bash
# logs 디렉토리가 없으면 생성
mkdir -p ~/Isaac-GR00T/logs

# 테스트 job 제출
sbatch test_finetune_dryrun.slurm
```

**확인 방법:**
```bash
# Job 상태 확인
squeue -u $USER

# Job ID 확인 후 로그 보기
tail -f ~/Isaac-GR00T/logs/test_<JOB_ID>.out
```

**이 테스트가 확인하는 것:**
- [ ] SLURM job이 정상적으로 시작됨
- [ ] $TMPDIR 사용 가능 및 충분한 공간
- [ ] 프로젝트 복사 성공
- [ ] Python 모듈 로드 성공
- [ ] GPU 인식 및 CUDA 작동
- [ ] Hugging Face CLI 작동
- [ ] rsync로 결과 복사 성공

---

## 🚀 실제 Fine-tuning 실행

### 실행 전 최종 확인
```bash
# 1. 가상환경에 필요한 패키지 설치 확인
source ~/Isaac-GR00T/gr00t/bin/activate
pip list | grep -E 'torch|huggingface'

# 2. Hugging Face 로그인 (처음 한 번만)
huggingface-cli login

# 3. logs 디렉토리 확인
ls -ld ~/Isaac-GR00T/logs
```

### Job 제출
```bash
cd ~/Isaac-GR00T
sbatch finetune_gr00t.slurm
```

제출 후 Job ID가 표시됩니다 (예: `Submitted batch job 12345`)

---

## 📊 학습 모니터링

### 방법 1: 모니터링 스크립트 사용
```bash
# Job ID를 확인 후 실행
./monitor_training.sh <JOB_ID>

# 예시:
./monitor_training.sh 12345
```

**제공 정보:**
- Job 상태 (PENDING, RUNNING, COMPLETED)
- 실행 노드 및 GPU 사용률
- 로그 파일 상태
- 학습 진행률 (step, loss)
- 에러 발생 여부
- 체크포인트 저장 현황

### 방법 2: 수동 모니터링

#### Job 상태 확인
```bash
# 내 모든 Job 보기
squeue -u $USER

# 특정 Job 상세 정보
scontrol show job <JOB_ID>
```

**상태 설명:**
- `PENDING (PD)`: 대기 중 (자원 할당 대기)
- `RUNNING (R)`: 실행 중
- `COMPLETING (CG)`: 종료 중
- `COMPLETED (CD)`: 완료
- `FAILED (F)`: 실패

#### 실시간 로그 보기
```bash
# Output 로그 (학습 진행 상황)
tail -f ~/Isaac-GR00T/logs/finetune_<JOB_ID>.out

# Error 로그 (문제 발생 시)
tail -f ~/Isaac-GR00T/logs/finetune_<JOB_ID>.err
```

#### GPU 사용률 확인
```bash
# 실시간 GPU 모니터링
watch -n 2 "srun --jobid=<JOB_ID> nvidia-smi"
```

**정상 상태:**
- GPU Utilization: 80-100%
- Memory Usage: 거의 전부 사용
- Temperature: 60-85°C

---

## 🔍 각 단계별 예상 시간 및 로그

### 1. 초기화 단계 (2-5분)
**로그 내용:**
```
========================================
Job ID: 12345
Node: gpu-node-01
Working directory: /tmp/slurm.12345.0
========================================
Copying project to scratch...
```

**확인:**
- Job ID와 노드 할당 확인
- 프로젝트 복사 성공

### 2. 데이터셋 다운로드 (10-30분)
**로그 내용:**
```
Downloading datasets to scratch...
Downloading dataset 1/3: recode-bimanual-red-block-basket-v2.1...
Downloading: 100%|██████████| 1.2G/1.2G
```

**확인:**
- 3개 데이터셋 모두 다운로드 완료
- "Datasets downloaded successfully!" 메시지

### 3. 환경 설정 (1-2분)
**로그 내용:**
```
========================================
Environment Info:
Python: Python 3.10.8
PyTorch: 2.x.x
CUDA available: True
GPU count: 4
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
========================================
```

**확인:**
- 4개 GPU 모두 인식
- CUDA 사용 가능

### 4. Fine-tuning 실행 (수 시간 ~ 20시간)
**로그 내용:**
```
Starting fine-tuning...
Step 0/10000 | Loss: 2.345 | LR: 0.0001
Step 100/10000 | Loss: 1.234 | LR: 0.0001
...
```

**확인:**
- Step 번호 증가
- Loss 값 감소 추세
- GPU 사용률 높음

**예상 시간:**
- 10,000 steps
- Step당 약 3-5초 (4 GPU 기준)
- 총 8-14시간 정도

### 5. 결과 복사 (5-10분)
**로그 내용:**
```
Copying results back to home directory...
sending incremental file list
checkpoint-1000/
checkpoint-1000/pytorch_model.bin
...
Training completed successfully!
Checkpoints saved to: /home/jxl2244/Isaac-GR00T/so101-bimanual-checkpoints/
```

**확인:**
- 체크포인트 파일들이 home으로 복사됨
- "Training completed successfully!" 메시지

---

## ⚠️ 문제 해결

### 문제 1: Job이 PENDING 상태에서 멈춤
**원인:** 자원 부족 (4-GPU 노드 대기 중)

**확인:**
```bash
squeue -u $USER -o "%.18i %.9P %.8T %.10M %.10l %.6D %R"
```

**해결:**
- `Reason` 컬럼 확인
- `Priority`: 우선순위 대기 → 기다림
- `Resources`: 자원 부족 → 시간대 변경 또는 GPU 개수 줄이기

### 문제 2: "hf: command not found"
**원인:** Hugging Face CLI 미설치

**해결:**
```bash
source ~/Isaac-GR00T/gr00t/bin/activate
pip install -U "huggingface_hub[cli]"
```

### 문제 3: 데이터셋 다운로드 실패
**원인:** Hugging Face 인증 필요

**해결:**
```bash
# 로그인 노드에서
source ~/Isaac-GR00T/gr00t/bin/activate
huggingface-cli login
# Token 입력 (https://huggingface.co/settings/tokens)
```

### 문제 4: GPU Out of Memory (OOM)
**원인:** Batch size가 GPU 메모리보다 큼

**해결 (finetune_gr00t.slurm 수정):**
```bash
# Line 93-94 수정
  --batch-size 4 \                           # 8 → 4로 줄임
  --gradient-accumulation-steps 8 \          # 4 → 8로 늘림 (효과적 batch size 유지)
```

### 문제 5: 학습이 멈춘 것 같음
**확인:**
```bash
# 로그 파일 마지막 수정 시간 확인
ls -lh ~/Isaac-GR00T/logs/finetune_<JOB_ID>.out

# 최근 10분 내 업데이트 확인
find ~/Isaac-GR00T/logs/ -name "finetune_*.out" -mmin -10

# GPU가 작동 중인지 확인
srun --jobid=<JOB_ID> nvidia-smi
```

**정상:** GPU 사용률 80-100%, 로그 파일이 계속 업데이트됨  
**문제:** GPU 사용률 0%, 로그 멈춤 → Job 취소 후 재시작

### 문제 6: Job이 너무 오래 걸림
**확인:**
```bash
# 현재 step 확인
grep -i "step" ~/Isaac-GR00T/logs/finetune_<JOB_ID>.out | tail -5

# Step당 시간 계산
# 예: Step 100까지 500초 = 5초/step
# 10,000 steps × 5초 = 50,000초 = 약 14시간
```

**참고:** 24시간 제한 내에 완료되도록 `--max-steps` 조정 가능

---

## 💾 결과 확인

### 체크포인트 위치
```bash
ls -lh ~/Isaac-GR00T/so101-bimanual-checkpoints/
```

**예상 파일:**
```
checkpoint-1000/
checkpoint-2000/
...
checkpoint-10000/
final_model/
```

### 체크포인트 크기
- 각 체크포인트: 약 5-10GB
- 전체: 50-100GB (10개 체크포인트 기준)

### 로그 보관
```bash
# 로그 파일 위치
~/Isaac-GR00T/logs/finetune_<JOB_ID>.out
~/Isaac-GR00T/logs/finetune_<JOB_ID>.err
```

---

## 📝 리소스 사용 통계 확인

### Job 완료 후
```bash
# 상세 통계 확인
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,Elapsed,CPUTime,MaxRSS,MaxVMSize

# GPU 통계 (있는 경우)
sacct -j <JOB_ID> --format=JobID,State,Elapsed,ReqGRES,AllocGRES
```

---

## 🔄 다음 실행 시 개선

### 1. 이미 다운로드한 데이터셋 재사용
스크립트를 수정하여 home 디렉토리에 데이터셋 캐시:

```bash
# finetune_gr00t.slurm의 33-50행 수정
# 첫 실행: 다운로드 후 home에 저장
# 이후 실행: home에서 scratch로 복사 (다운로드보다 빠름)

if [ -d "$HOME/Isaac-GR00T/demo_data_cache" ]; then
    echo "Copying cached datasets from home..."
    cp -r $HOME/Isaac-GR00T/demo_data_cache ./demo_data
else
    echo "Downloading datasets..."
    # ... 기존 다운로드 코드 ...
    # 다운로드 후 캐시 저장
    cp -r ./demo_data $HOME/Isaac-GR00T/demo_data_cache
fi
```

### 2. 체크포인트 중간 저장
긴 학습 시 중간 체크포인트를 주기적으로 home으로 복사:

```bash
# 학습 중 백그라운드로 주기적 복사 (선택적)
# fine-tuning 스크립트에서 --save-steps 1000 설정 확인
```

---

## 📞 추가 도움

### HPC 지원 센터
- Email: hpc-support@your-institution.edu
- 문서: https://hpc.your-institution.edu/docs

### 유용한 SLURM 명령어
```bash
squeue -u $USER           # 내 Job 목록
scontrol show job <ID>    # Job 상세 정보
scancel <ID>              # Job 취소
sinfo -p markov_gpu       # 파티션 상태
seff <ID>                 # Job 효율성 분석 (완료 후)
```

---

## ✨ 요약

**실행 순서:**
1. `./test_setup.sh` → 기본 환경 확인
2. `sbatch test_finetune_dryrun.slurm` → SLURM 환경 테스트
3. `sbatch finetune_gr00t.slurm` → 실제 fine-tuning
4. `./monitor_training.sh <JOB_ID>` → 진행 상황 모니터링

**정상 작동 시 타임라인:**
- 0-5분: 초기화 및 프로젝트 복사
- 5-35분: 데이터셋 다운로드
- 35분-14시간: Fine-tuning 실행
- 14-14.5시간: 결과 복사
- 완료!

**문제 발생 시:**
- Error 로그 확인: `tail ~/Isaac-GR00T/logs/finetune_<JOB_ID>.err`
- 이 문서의 "문제 해결" 섹션 참조
- HPC 지원 센터 문의

Good luck! 🚀

