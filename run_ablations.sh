#!/bin/bash

# run_ablations.sh
# Script to run all FreqDINO ablation experiments sequentially

set -e  # Exit on error is disabled, we handle errors per experiment

echo "=========================================="
echo "FreqDINO Ablation Study Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Configuration
GPU_ID="3"
EPOCHS=20
BATCH_SIZE=32
LOG_FILE="ablation_experiments_$(date +%Y%m%d_%H%M%S).log"

# Create master log file
touch "$LOG_FILE"

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to run experiment with error handling
run_experiment() {
    local model_name=$1
    local run_name=$2
    
    log_message ""
    log_message "=========================================="
    log_message "Starting: $run_name"
    log_message "Model: $model_name"
    log_message "Time: $(date)"
    log_message "=========================================="
    
    if python train.py \
        --model "$model_name" \
        --run_name "$run_name" \
        --gpu "$GPU_ID" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"; then
        
        log_message ""
        log_message "SUCCESS: $run_name completed"
        log_message "Completion time: $(date)"
        log_message ""
        return 0
    else
        log_message ""
        log_message "ERROR: $run_name failed with exit code $?"
        log_message "Failure time: $(date)"
        log_message "Continuing with next experiment..."
        log_message ""
        return 1
    fi
}

# Track experiment results
declare -a success_experiments
declare -a failed_experiments

#experiment 0: Ablation with contrastive loss
if run_experiment "ablation0" "freq_dino_ablation0_contrastive"; then
    success_experiments+=("ablation0")
else
    failed_experiments+=("ablation0")
fi


# Experiment 1: Baseline (Full Model)
if run_experiment "baseline" "freq_dino_baseline"; then
    success_experiments+=("baseline")
else
    failed_experiments+=("baseline")
fi

# Experiment 2: Ablation 1 - No Noise Bank
if run_experiment "ablation1" "freq_dino_ablation1_no_noise"; then
    success_experiments+=("ablation1")
else
    failed_experiments+=("ablation1")
fi

# Experiment 3: Ablation 2 - w/o BC-PCA
if run_experiment "ablation2" "freq_dino_ablation2_no_bcpca"; then
    success_experiments+=("ablation2")
else
    failed_experiments+=("ablation2")
fi

# Experiment 4: Ablation 3 - w/o BC-PCA and CLIP
if run_experiment "ablation3" "freq_dino_ablation3_no_bcpca_clip"; then
    success_experiments+=("ablation3")
else
    failed_experiments+=("ablation3")
fi

# Experiment 5: Ablation 4 - No DINO Patches
if run_experiment "ablation4" "freq_dino_ablation4_no_dino_patches"; then
    success_experiments+=("ablation4")
else
    failed_experiments+=("ablation4")
fi

# Experiment 6: Ablation 5 - Only DINO Features
if run_experiment "ablation5" "freq_dino_ablation5_only_dino"; then
    success_experiments+=("ablation5")
else
    failed_experiments+=("ablation5")
fi

# Summary Report
log_message ""
log_message "=========================================="
log_message "ABLATION STUDY SUMMARY"
log_message "=========================================="
log_message "End time: $(date)"
log_message ""
log_message "Total experiments: 6"
log_message "Successful: ${#success_experiments[@]}"
log_message "Failed: ${#failed_experiments[@]}"
log_message ""

if [ ${#success_experiments[@]} -gt 0 ]; then
    log_message "Successful experiments:"
    for exp in "${success_experiments[@]}"; do
        log_message "  ✓ $exp"
    done
    log_message ""
fi

if [ ${#failed_experiments[@]} -gt 0 ]; then
    log_message "Failed experiments:"
    for exp in "${failed_experiments[@]}"; do
        log_message "  ✗ $exp"
    done
    log_message ""
fi

log_message "Full log saved to: $LOG_FILE"
log_message "Individual experiment logs in: ./experiments/"
log_message "=========================================="

# Exit with error if any experiments failed
if [ ${#failed_experiments[@]} -gt 0 ]; then
    exit 1
else
    log_message "All experiments completed successfully!"
    exit 0
fi

# chmod +x run_ablations.sh
# ./run_ablations.sh