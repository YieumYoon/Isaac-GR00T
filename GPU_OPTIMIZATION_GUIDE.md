# GR00T Fine-tuning GPU Optimization Guide

## Cluster GPU Resources Overview

Your Markov GPU cluster has the following resources:

| Node Type | GPUs | CPUs | RAM | Best For |
|-----------|------|------|-----|----------|
| classt01-19 | 2 | 20 | 128GB | Small-medium jobs |
| classt21-22, 25 | 4 | 64-96 | 257-386GB | Standard multi-GPU training |
| classt23-24 | **14** | 24-48 | 515GB | **Maximum GPU jobs** |

## Optimization Summary

### Original Configuration (Single GPU)
- **GPUs**: 1
- **Batch size per GPU**: 4
- **Gradient accumulation**: 8
- **Effective batch size**: 4 × 1 × 8 = **32**
- **Training speed**: ~X iterations/sec

### Optimized Configuration (4 GPUs) ⭐ RECOMMENDED
- **GPUs**: 4
- **Batch size per GPU**: 8
- **Gradient accumulation**: 4
- **Effective batch size**: 8 × 4 × 4 = **128** (4x throughput)
- **Training speed**: ~4X iterations/sec
- **File**: `finetune_gr00t.slurm`

### Maximum Configuration (8 GPUs)
- **GPUs**: 8
- **Batch size per GPU**: 12
- **Gradient accumulation**: 2
- **Effective batch size**: 12 × 8 × 2 = **192** (6x throughput)
- **Training speed**: ~8X iterations/sec
- **File**: `finetune_gr00t_max_gpu.slurm`

## Key Optimizations Applied

### 1. Multi-GPU Training
```bash
#SBATCH --gres=gpu:4              # Request 4 GPUs
--num-gpus 4                       # Tell training script to use 4 GPUs
```

### 2. Resource Scaling
- **CPUs**: 8 per GPU (4 GPUs × 8 = 32 CPUs)
- **Memory**: 64GB per GPU (4 GPUs × 64GB = 256GB)
- **Exclusive node**: Ensures no resource contention

### 3. Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3              # Specify GPU IDs
export OMP_NUM_THREADS=8                          # CPU threads per process
export NCCL_DEBUG=INFO                            # Monitor GPU communication
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better memory management
```

### 4. Batch Size Optimization
- Increased per-GPU batch size from 4 to 8 (or 12 for 8-GPU config)
- Reduced gradient accumulation steps (still maintains large effective batch)
- More efficient GPU utilization

### 5. Data Loading
- Uses scratch space ($TMPDIR) for fast I/O
- Copies datasets to local node storage
- Reduces network bottleneck

## Usage Instructions

### Option 1: Standard 4-GPU Training (Recommended)
```bash
sbatch finetune_gr00t.slurm
```
**Use when**: You want good speedup with high availability

### Option 2: Maximum 8-GPU Training
```bash
sbatch finetune_gr00t_max_gpu.slurm
```
**Use when**: You need maximum speed and the high-GPU nodes are available

### Option 3: Custom GPU Count
Edit the SLURM parameters based on your needs:

For **2 GPUs**:
```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# In training command:
--num-gpus 2 --batch-size 6 --gradient-accumulation-steps 5
# Effective batch size: 6 × 2 × 5 = 60
```

For **14 GPUs** (Maximum):
```bash
#SBATCH --gres=gpu:14
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --nodelist=classt23  # or classt24
# In training command:
--num-gpus 14 --batch-size 8 --gradient-accumulation-steps 1
# Effective batch size: 8 × 14 × 1 = 112
```

## Monitoring GPU Usage

### During Job Execution
```bash
# SSH to the node running your job
ssh classt21  # Replace with your actual node

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### Check Job Status
```bash
# View your job
squeue -u $USER

# View detailed job info
scontrol show job <JOBID>

# View job output in real-time
tail -f logs/finetune_<JOBID>.out
```

## Expected Performance Improvements

| Configuration | GPUs | Time to 10k Steps | Relative Speed | GPU Utilization |
|--------------|------|-------------------|----------------|-----------------|
| Original | 1 | ~40 hours | 1x | 60-70% |
| Optimized (4 GPU) | 4 | ~10 hours | 4x | 85-95% |
| Maximum (8 GPU) | 8 | ~5 hours | 8x | 85-95% |

*Actual times depend on GPU model and dataset size*

## Troubleshooting

### Out of Memory (OOM) Errors
If you get CUDA OOM errors, reduce batch size:
```bash
--batch-size 4  # Reduce from 8
--gradient-accumulation-steps 8  # Increase to maintain effective batch size
```

### Slow Data Loading
If GPU utilization is low (<70%), increase data workers:
```bash
--num-workers 8  # Add this parameter if available
```

### NCCL/Communication Issues
If GPUs can't communicate:
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer if causing issues
```

## Advanced: Hyperparameter Tuning for Multi-GPU

When using multiple GPUs, you may want to adjust:

1. **Learning Rate**: Scale with effective batch size
   - Original: lr=1e-4 for batch_size=32
   - 4 GPUs: lr=2e-4 for batch_size=128
   - 8 GPUs: lr=3e-4 for batch_size=192

2. **Warmup Steps**: Scale with GPU count
   - Original: warmup=1000
   - 4 GPUs: warmup=500 (converges faster)
   - 8 GPUs: warmup=250

3. **Gradient Clipping**: May need adjustment
   - Monitor gradient norms in logs
   - Adjust if seeing instability

## Cost-Benefit Analysis

| Metric | 1 GPU | 4 GPU | 8 GPU |
|--------|-------|-------|-------|
| Training Time | 40h | 10h | 5h |
| GPU-Hours Used | 40 | 40 | 40 |
| Time to Results | Longest | Good | Best |
| Resource Availability | High | Medium | Low |
| **Recommendation** | Baseline | ⭐ Best | Fastest |

**Recommendation**: Use the 4-GPU configuration for the best balance of speed and resource availability.

## Validation

After optimizing, verify your changes didn't affect model quality:

1. **Compare Training Loss**: Should be similar trajectory
2. **Check Validation Metrics**: Should match or improve
3. **Test Final Model**: Verify robot performance

## Additional Resources

- PyTorch Distributed Training: https://pytorch.org/tutorials/beginner/dist_overview.html
- NCCL Best Practices: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- SLURM GPU Documentation: https://slurm.schedmd.com/gres.html

## Quick Reference Commands

```bash
# Submit 4-GPU job (recommended)
sbatch finetune_gr00t.slurm

# Submit 8-GPU job (maximum)
sbatch finetune_gr00t_max_gpu.slurm

# Check available GPUs
sinfo -p markov_gpu -o "%N %G %c %m %C"

# Monitor your job
watch squeue -u $USER

# Cancel job if needed
scancel <JOBID>

# View logs
tail -f logs/finetune_*.out
```

---

**Last Updated**: October 2025
**For Questions**: Contact your cluster administrator or check SLURM documentation

