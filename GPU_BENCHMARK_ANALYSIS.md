# 🚀 Isaac-GR00T: 최고 성능 학습 전략

## 📊 서버 GPU 성능 분석

### **사용 가능한 GPU 목록**

| GPU 모델 | 개수 | VRAM | FP32 TFLOPS | FP16 TFLOPS | 노드 | 가용성 |
|----------|------|------|-------------|-------------|------|--------|
| **RTX 4090** | 4 | 24GB | 82.6 | 165.2 | classt25 | ⚠️ 경쟁 높음 |
| **H100 NVL** | 2 | 94GB | 51.2 | 1979 | classt23-24 | ⚠️ MIG 모드 (11GB) |
| **L40S** | 6×2 | 48GB | 91.6 | 183.2 | classt21-22 | ✅ 대기 가능 |
| **RTX 2080 Ti** | 4×15 | 11GB | 13.4 | 26.9 | classt01-18 | ✅ 많이 가용 |
| **RTX 4070** | 4 | 12GB | 29.1 | 58.2 | classt06 | ✅ 가용 |

### **성능 순위 (FP16 기준 - AI 학습 최적)**

```
1. 🥇 H100 NVL: 1979 TFLOPS (하지만 MIG로 11GB만 사용 가능)
2. 🥈 L40S: 183.2 TFLOPS × 6 = 1099 TFLOPS (단일 노드)
3. 🥉 RTX 4090: 165.2 TFLOPS × 4 = 660 TFLOPS (단일 노드)
4. RTX 4070: 58.2 TFLOPS × 4 = 232 TFLOPS
5. RTX 2080 Ti: 26.9 TFLOPS × 4 = 107 TFLOPS
```

---

## 🎯 최적 학습 전략

### **Strategy 1: 즉시 시작 (권장 ⭐)**

**목표**: 대기 시간 최소화, 빠른 반복

```bash
노드: classt25 (RTX 4090 × 4)
성능: 660 TFLOPS (FP16)
메모리: 96GB 총
Batch size: 8-12 per GPU
대기 시간: 낮음-중간
```

**설정:**
```bash
#SBATCH --gres=gpu:gpu4090:4
#SBATCH --nodelist=classt25
--batch-size 10
--gradient-accumulation-steps 3
# Effective: 10 × 4 × 3 = 120
```

**예상 시간**: 10,000 steps ≈ 10-12시간

---

### **Strategy 2: 최고 성능 (6 GPU)**

**목표**: 최대 처리량, 빠른 완료

```bash
노드: classt21 또는 classt22 (L40S × 6)
성능: 1099 TFLOPS (FP16)
메모리: 288GB 총
Batch size: 10-14 per GPU
대기 시간: 중간-높음
```

**설정:**
```bash
#SBATCH --gres=gpu:gpul40s:6
#SBATCH --nodelist=classt21
--batch-size 12
--gradient-accumulation-steps 2
# Effective: 12 × 6 × 2 = 144
```

**예상 시간**: 10,000 steps ≈ 7-9시간

---

### **Strategy 3: 경제적 (RTX 2080 Ti)**

**목표**: 항상 사용 가능, 비용 효율

```bash
노드: classt01-18 (RTX 2080 Ti × 4)
성능: 107 TFLOPS (FP16)
메모리: 44GB 총
Batch size: 4-6 per GPU
대기 시간: 거의 없음
```

**설정:**
```bash
#SBATCH --gres=gpu:gpu2080:4
--batch-size 5
--gradient-accumulation-steps 6
# Effective: 5 × 4 × 6 = 120
```

**예상 시간**: 10,000 steps ≈ 18-24시간

---

## 📐 Batch Size 최적화 계산

### **GPU 메모리별 권장 Batch Size**

| GPU | VRAM | 보수적 | 권장 | 공격적 | 주의사항 |
|-----|------|--------|------|--------|----------|
| RTX 2080 Ti | 11GB | 2 | 4 | 6 | OOM 주의 |
| RTX 4070 | 12GB | 3 | 5 | 7 | - |
| RTX 4090 | 24GB | 6 | 10 | 14 | 최적 |
| L40S | 48GB | 12 | 16 | 20 | 최고 |
| H100 (MIG) | 11GB | 2 | 4 | 6 | MIG 제한 |

### **계산 공식**

```python
# GR00T fine-tuning 메모리 추정
base_model_memory = 5 GB  # Model weights
per_sample_memory = 1.2 GB  # Gradients + activations
overhead = 2 GB  # CUDA kernels, etc.

max_batch_size = floor((GPU_VRAM - base_model_memory - overhead) / per_sample_memory)
```

### **Effective Batch Size 유지**

```
Target Effective Batch Size: 120-144

Formula: batch_size × num_gpus × gradient_accumulation_steps = 120-144

Examples:
- 4 GPU, batch=10: 10 × 4 × 3 = 120 ✅
- 6 GPU, batch=12: 12 × 6 × 2 = 144 ✅
- 4 GPU, batch=5:  5 × 4 × 6 = 120 ✅
```

---

## ⚡ 성능 최적화 팁

### **1. Mixed Precision Training**

```python
# 이미 스크립트에 포함되어 있을 가능성 높음
fp16=True  # 2배 빠름, 메모리 절반
```

### **2. Gradient Checkpointing**

```python
# 메모리 부족 시
gradient_checkpointing=True  # 메모리 50% 감소, 속도 20% 감소
```

### **3. DataLoader 최적화**

```python
num_workers=8  # CPU 코어 활용
pin_memory=True  # GPU 전송 빠름
prefetch_factor=2  # 미리 로드
```

### **4. 컴파일 최적화 (PyTorch 2.0+)**

```python
torch.compile()  # 30-40% 속도 향상 (초기 컴파일 시간 필요)
```

---

## 🎯 최종 추천 전략

### **상황별 최적 선택**

#### **🏃 빠른 테스트/디버깅**
```bash
RTX 2080 Ti × 4 (항상 가용)
--batch-size 4
--max-steps 100
실행 시간: ~30분
```

#### **⚖️ 균형잡힌 학습 (추천)**
```bash
RTX 4090 × 4 (대기 시간 낮음)
--batch-size 10
--max-steps 10000
실행 시간: ~10시간
```

#### **🚀 최고 성능**
```bash
L40S × 6 (대기 가능하면)
--batch-size 12
--max-steps 10000
실행 시간: ~7시간
```

---

## 📋 실행 체크리스트

### **Job 제출 전**

- [ ] 로그 디렉토리 생성: `mkdir -p ~/Isaac-GR00T/logs`
- [ ] Hugging Face 로그인: `huggingface-cli login`
- [ ] GPU 가용성 확인: `sinfo -p markov_gpu`
- [ ] 할당량 확인: `quota -s`

### **Job 제출**

```bash
# Strategy 1: 4090 (권장)
sbatch finetune_gr00t.slurm

# Strategy 2: L40S (최고 성능)
sbatch --nodelist=classt21 --gres=gpu:gpul40s:6 finetune_gr00t_l40s.slurm

# Strategy 3: 2080 Ti (항상 가용)
sbatch --gres=gpu:gpu2080:4 finetune_gr00t_2080.slurm
```

### **모니터링**

```bash
# Job 상태
squeue -u $USER

# 실시간 로그
tail -f ~/Isaac-GR00T/logs/finetune_<JOB_ID>.out

# GPU 사용률
srun --jobid=<JOB_ID> nvidia-smi
```

---

## 💾 저장 공간 관리

### **디스크 사용량**

```bash
Home 할당량: 23.8GB
현재 사용: ~1GB (최적화 후)

예상 체크포인트 크기:
- 각 checkpoint: 5-8GB
- 10개 checkpoint: 50-80GB
- 최종 모델: 5-8GB
```

### **저장 전략**

```bash
# 체크포인트는 scratch에 저장되고, 완료 후 home으로 복사됨
# 중간 체크포인트 선택적 저장 (공간 절약)
--save-strategy "steps"
--save-steps 2000  # 5개만 저장
```

---

## 🔬 실험 로그

### **Experiment 1: Baseline**
```
Date: [TBD]
GPUs: RTX 4090 × 4
Batch Size: 8
Steps: 10000
Time: [TBD]
Final Loss: [TBD]
```

### **Experiment 2: Optimized**
```
Date: [TBD]
GPUs: [TBD]
Batch Size: [TBD]
Steps: 10000
Time: [TBD]
Final Loss: [TBD]
```

---

## 📞 문제 해결

### **일반적인 이슈**

1. **OOM (Out of Memory)**
   - Batch size를 절반으로 줄이기
   - Gradient accumulation 2배 늘리기
   - Gradient checkpointing 활성화

2. **느린 데이터 로딩**
   - `num_workers` 증가 (8-16)
   - `prefetch_factor` 증가 (2-4)
   - 데이터셋 크기 확인

3. **GPU 사용률 낮음 (<80%)**
   - Batch size 증가
   - DataLoader workers 증가
   - 병목 지점 프로파일링

4. **Hang/Timeout**
   - NCCL 설정 확인
   - 네트워크 문제 (멀티 노드)
   - Timeout 값 증가

---

## 🎓 참고 자료

- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [Hugging Face Training Tips](https://huggingface.co/docs/transformers/performance)

---

**Created**: 2025-10-25  
**Last Updated**: 2025-10-25  
**Status**: Active  

