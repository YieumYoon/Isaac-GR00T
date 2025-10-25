# 🚀 Isaac-GR00T 최종 학습 전략

## 📊 실제 GPU 구성 (정확한 조사 결과)

### **사용 가능한 GPU 노드**

| 노드 | GPU 타입 | GPU 개수 | GPU 메모리 | CPU | RAM | Feature | 상태 |
|------|----------|----------|------------|-----|-----|---------|------|
| **classt21** | L40S | 4 | 48GB | 96 | 257GB | `gpul40s` | ✅ IDLE |
| **classt22** | L40S | 4 | 48GB | 96 | 257GB | `gpul40s` | ✅ IDLE |
| **classt25** | RTX 4090 | 4 | 24GB | 64 | 386GB | `gpu4090` | ✅ IDLE |
| **classt23** | H100 NVL | 2 (14 MIG) | 94GB | 48 | 514GB | `gpu2h100` | ✅ IDLE |
| **classt24** | H100 NVL | 2 (14 MIG) | 94GB | 24 | 514GB | `gpu2h100` | ✅ IDLE |
| classt01-18 | RTX 2080 Ti | 2 | 11GB | 20 | 128GB | `gpu2080` | ✅ 많음 |

### **핵심 발견사항**

1. **GPU 타입 지정 방법**:
   - ❌ 잘못: `--gres=gpu:gpul40s:4`
   - ✅ 올바름: `--gres=gpu:4 --constraint=gpul40s`
   - ✅ 또는: `--gres=gpu:4 -C gpul40s`

2. **실제 GPU 개수**:
   - L40S: 노드당 **4개** (6개 아님!)
   - RTX 4090: 노드당 **4개**
   - H100: 물리적 2개, MIG로 14개 인스턴스

---

## 🎯 최적 전략 (성능 순위)

### **🥇 전략 1: L40S (최고 성능 + 메모리)**

```bash
노드: classt21 또는 classt22
GPU: L40S × 4 (48GB each = 192GB total)
성능: 183 TFLOPS (FP16) × 4 = 732 TFLOPS
```

**SLURM 설정:**
```bash
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpul40s
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
```

**학습 설정:**
```bash
--batch-size 16        # L40S 48GB 최대 활용
--gradient-accumulation-steps 2
# Effective: 16 × 4 × 2 = 128
```

**예상 시간**: 10,000 steps ≈ **7-9시간**

---

### **🥈 전략 2: RTX 4090 (균형)**

```bash
노드: classt25
GPU: RTX 4090 × 4 (24GB each = 96GB total)
성능: 165 TFLOPS (FP16) × 4 = 660 TFLOPS
```

**SLURM 설정:**
```bash
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu4090
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
```

**학습 설정:**
```bash
--batch-size 10
--gradient-accumulation-steps 3
# Effective: 10 × 4 × 3 = 120
```

**예상 시간**: 10,000 steps ≈ **10-12시간**

---

### **🥉 전략 3: RTX 2080 Ti (항상 사용 가능)**

```bash
노드: classt01-18 (많음)
GPU: RTX 2080 Ti × 2 (11GB each = 22GB total)
성능: 27 TFLOPS (FP16) × 2 = 54 TFLOPS
```

**SLURM 설정:**
```bash
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu2080
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
```

**학습 설정:**
```bash
--batch-size 4
--gradient-accumulation-steps 10
# Effective: 4 × 2 × 10 = 80
```

**예상 시간**: 10,000 steps ≈ **18-24시간**

---

## 💡 최종 권장 사항

### **즉시 시작하려면: L40S (추천!)**

L40S가 **IDLE 상태**이고 **48GB 메모리**로 가장 큰 batch size 사용 가능!

```bash
sbatch finetune_gr00t_L40S.slurm
```

### **대기 시간 최소화: 4090**

4090도 IDLE이고 충분히 빠름:

```bash
sbatch finetune_gr00t_4090.slurm
```

---

## 📋 스크립트 수정 사항

### **중요한 수정**

1. **GPU 타입 지정 방식** (Feature 사용):
   ```bash
   # 잘못된 방식
   #SBATCH --gres=gpu:gpul40s:4  ❌
   
   # 올바른 방식
   #SBATCH --gres=gpu:4
   #SBATCH --constraint=gpul40s  ✅
   ```

2. **huggingface_hub 설치 순서**:
   ```bash
   # 모듈 로드 → venv 생성 → hf CLI 설치 → 데이터 다운로드
   module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.6.0
   python -m venv venv
   source venv/bin/activate
   pip install "huggingface_hub[cli]"  # 먼저!
   hf download ...  # 그 다음
   ```

3. **gr00t 패키지 설치**:
   ```bash
   # 데이터 다운로드 후
   pip install -e .[base]
   ```

---

## ⚡ 최적화된 스크립트 템플릿

완전히 작동하는 최종 버전을 제공합니다!

