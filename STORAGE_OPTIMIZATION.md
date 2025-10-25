# Storage Optimization for GR00T Fine-tuning

## Problem
Datasets are too large for home directory storage quota (home directory limit: ~15GB based on your du output).

## Solution
Modified all SLURM scripts to download datasets **directly to compute node temporary storage** (`$TMPDIR`) instead of copying from home directory.

## How It Works

### 1. Job Submission Flow
```bash
# Submit job as usual
sbatch finetune_gr00t.slurm  # or any other variant
```

### 2. What Happens During Job Execution

**On Compute Node (in $TMPDIR):**
1. ✅ Copy project code to scratch (small, <1GB)
2. ✅ Download datasets directly to scratch (large, several GB)
3. ✅ Run training with datasets in scratch
4. ✅ Copy **only checkpoints** back to home directory
5. ✅ Datasets are automatically deleted when job ends (scratch is cleaned up)

### 3. Storage Usage

| Location | Before | After |
|----------|---------|--------|
| Home directory | ~15GB (over quota) | ~3-5GB (project code + checkpoints only) |
| Compute node scratch | 0GB | Temporary: 10-20GB during job, 0GB after |

## Benefits

✅ **No home directory quota issues** - datasets never stored in home  
✅ **Faster I/O** - scratch storage is typically faster than networked home  
✅ **Automatic cleanup** - datasets deleted when job completes  
✅ **Fresh data** - always downloads latest version from Hugging Face  

## Modified Files

- `finetune_gr00t.slurm` (4 GPUs)
- `finetune_gr00t_14gpu.slurm` (14 GPUs)
- `finetune_gr00t_max_gpu.slurm` (8 GPUs)
- `setup.md` (updated documentation)

## What Gets Saved to Home Directory

Only the important results are copied back:
- **Checkpoints:** `~/Isaac-GR00T/so101-bimanual-checkpoints/`
- **Logs:** `~/Isaac-GR00T/logs/`

## Customization for Your Own Dataset

If you want to use your own custom dataset, modify the SLURM script's download section:

```bash
# Replace the existing download commands with your dataset
hf download --repo-type dataset YOUR_USERNAME/YOUR_DATASET \
    --local-dir ./demo_data/YOUR_DATASET_NAME
```

Then update the `--dataset-path` argument in the training command:

```bash
python scripts/gr00t_finetune.py \
  --dataset-path ./demo_data/YOUR_DATASET_NAME/ \
  ...
```

## Troubleshooting

### If download fails during job:
- Check Hugging Face credentials: `echo $HF_TOKEN`
- Verify you have access to the datasets on Hugging Face
- Check network connectivity from compute nodes

### If job runs out of scratch space:
- Check available scratch space: `df -h $TMPDIR`
- Request nodes with larger scratch storage
- Reduce number of datasets downloaded simultaneously

### To monitor scratch space usage during job:
Add to SLURM script:
```bash
echo "Scratch space usage:"
df -h $TMPDIR
du -sh demo_data/
```

## Notes

- ⚠️ Network speed on compute nodes will affect download time
- ⚠️ First run of each job will take longer due to download
- ⚠️ Datasets are NOT cached between jobs (re-downloaded each time)
- ℹ️ If you need to cache datasets, consider using a shared scratch filesystem (e.g., `/scratch` or `/project`)

