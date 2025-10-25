#!/bin/bash
# Quick script to check GPU availability and recommend configuration

echo "========================================="
echo "Markov GPU Cluster - Resource Check"
echo "========================================="
echo ""

echo "Current Queue Status:"
squeue -p markov_gpu -o "%.10i %.9P %.20j %.8u %.2t %.10M %D %b" | head -20
echo ""

echo "Available GPU Nodes:"
sinfo -p markov_gpu -o "%N %G %C %m %E" | grep -v "NODELIST"
echo ""

echo "Detailed Node Status:"
echo "----------------------------------------"
sinfo -p markov_gpu -N -o "%-12N %-15G %-20C %-12m %-10e" | grep classt
echo ""

# Count idle GPUs
IDLE_2GPU=$(sinfo -p markov_gpu -h -N -o "%N %G %T" | grep -E "classt0[1-9]|classt1[0-9]" | grep "idle" | wc -l)
IDLE_4GPU=$(sinfo -p markov_gpu -h -N -o "%N %G %T" | grep -E "classt2[1-2]|classt25" | grep "idle" | wc -l)
IDLE_14GPU=$(sinfo -p markov_gpu -h -N -o "%N %G %T" | grep -E "classt2[3-4]" | grep "idle" | wc -l)

echo "========================================="
echo "GPU Availability Summary:"
echo "========================================="
echo "Idle 2-GPU nodes:  $IDLE_2GPU"
echo "Idle 4-GPU nodes:  $IDLE_4GPU"
echo "Idle 14-GPU nodes: $IDLE_14GPU"
echo ""

echo "========================================="
echo "Recommended Configuration:"
echo "========================================="

if [ $IDLE_14GPU -gt 0 ]; then
    echo "✓ 14-GPU nodes available!"
    echo "  → Use: sbatch finetune_gr00t_max_gpu.slurm"
    echo "  → Edit script to use --gres=gpu:14 for maximum speed"
    echo ""
elif [ $IDLE_4GPU -gt 0 ]; then
    echo "✓ 4-GPU nodes available! (RECOMMENDED)"
    echo "  → Use: sbatch finetune_gr00t_max_gpu.slurm"
    echo "  → Or use: sbatch finetune_gr00t.slurm (already configured for 4 GPUs)"
    echo ""
elif [ $IDLE_2GPU -gt 0 ]; then
    echo "✓ 2-GPU nodes available"
    echo "  → Edit finetune_gr00t.slurm:"
    echo "    Change --gres=gpu:2, --cpus-per-task=16, --mem=128G"
    echo "    Change --num-gpus 2, --batch-size 6, --gradient-accumulation-steps 5"
    echo ""
else
    echo "⚠ All nodes busy. Your job will queue."
    echo "  → Submit with: sbatch finetune_gr00t.slurm"
    echo "  → Check queue with: squeue -p markov_gpu"
    echo ""
fi

echo "========================================="
echo "Quick Commands:"
echo "========================================="
echo "Submit 4-GPU job:    sbatch finetune_gr00t.slurm"
echo "Submit 8-GPU job:    sbatch finetune_gr00t_max_gpu.slurm"
echo "Check your jobs:     squeue -u \$USER"
echo "Cancel job:          scancel <JOBID>"
echo "View logs:           tail -f logs/finetune_*.out"
echo "========================================="

