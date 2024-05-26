#!/bin/bash

JOBS=("fid" "clip")
DIRS=("")
NUMS=("999")

GPU_START=0  # GPU starting ID
GPU_END=3    # GPU ending ID

JOB_NUM=1

# Helper function to check GPU availability
check_gpu() {
    GPU_ID=$1
    # Check if the number of processes is less than 2
    [[ $(nvidia-smi -i $GPU_ID --query-compute-apps=pid --format=csv,noheader | wc -l) -lt $JOB_NUM ]]
}

# Main loop
for job in "${JOBS[@]}"; do
    for dir in "${DIRS[@]}"; do 
        for num in "${NUMS[@]}"; do 
            # Find an available GPU
            GPU=-1
            while [[ $GPU -lt $GPU_START ]]; do
                for (( i=$GPU_START; i<=$GPU_END; i++ )); do
                    if check_gpu $i; then
                        GPU=$i
                        break
                    fi
                done
                # If no GPU with less than two processes found, sleep for a while before checking again
                if [[ $GPU -lt $GPU_START ]]; then
                    sleep 15
                fi
            done

            if [[ $job == "fid" ]]; then
                suffix="visualizations_fid_10k"
            elif [[ $job == "clip" ]]; then
                suffix="visualizations_fid_10k"
            fi


            # Run the job on the available GPU
            CUDA_VISIBLE_DEVICES=$GPU python train-scripts/img_retain_eval.py \
                --gen_imgs_path "${dir}_${num}_${suffix}/SD-v1-4" \
                --job $job &

            # Sleep for a bit to make sure the job starts
            sleep 15
        done
    done
done
# Wait for all background jobs to complete
wait
echo "All jobs completed."
