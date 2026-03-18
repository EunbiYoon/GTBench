#!/bin/bash
#SBATCH --job-name=bos_traj_gen
#SBATCH --output=outputs/logs/bos_%j.out
#SBATCH --error=outputs/logs/bos_%j.err
#SBATCH --partition=gpu-preempt         # change to your partition (gpu, gpu-long, etc.)
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00                 # 6 hours should be plenty for default config

# ── 1. Environment ──────────────────────────────────────────────────────────
module load conda/latest
conda activate my_env

# Make sure output dirs exist
mkdir -p outputs/logs outputs/trajectories outputs/dpo_pairs

# ── 2. Point to your project directory ──────────────────────────────────────
cd $SLURM_SUBMIT_DIR     # assumes you sbatch from inside the project folder

# ── 3. (Optional) set HuggingFace cache to scratch to avoid home quota hits ─
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# ── 4. Generation ───────────────────────────────────────────────────────────
echo "=== Starting trajectory generation === $(date)"
python generate.py --config config.yaml --seed 42

# Capture the most recently created trajectories file
TRAJ_FILE=$(ls -t outputs/trajectories/trajectories_*.jsonl | head -1)
echo "Trajectories file: $TRAJ_FILE"

# ── 5. Pairing ──────────────────────────────────────────────────────────────
echo "=== Starting DPO pairing === $(date)"
python score_and_pair.py \
    --input "$TRAJ_FILE" \
    --config config.yaml \
    --show-sample

echo "=== Done === $(date)"
