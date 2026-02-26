#!/bin/bash

USER_ROOT=${USER_ROOT:-$(dirname $(dirname "$PWD"))}  # ../../

PROJ_ROOT=${PROJ_ROOT:-"${USER_ROOT}/projects/JavisGPT"}
DATA_ROOT=${DATA_ROOT:-"${USER_ROOT}/datasets/JavisGPT"}
WEIGHT_ROOT=${WEIGHT_ROOT:-"${USER_ROOT}/weights"}
EVAL_DATA_ROOT="${DATA_ROOT}/eval/JavisUnd-Eval"
OUTPUT_DIR=${OUTPUT_DIR:-"results/javisgpt_v10/av_und"}

NUM_FRAMES=16
FRAME_WIDTH=588  # 448(4:3) or 588(16:9)
FRAME_HEIGHT=336
AUDIO_SR=16000
MAX_AUDIO_LENGTH_S=30
AVSYNC_MODE="merge"

export BASE_ARCH="Qwen2_5_VL"
CKPT_PATH="${WEIGHT_ROOT}/pretrained/mllm/Qwen2.5-VL-7B-Instruct"
MODEL_MAX_LENGTH=131072
BEATS_PATH="${WEIGHT_ROOT}/pretrained/mllm/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

MODEL_PATH=${MODEL_PATH:-"${PROJ_ROOT}/runs/javisgpt_stage3_mm_insttune"}
ALL_PROJ_PATH="${MODEL_PATH}/mm_proj_all.bin"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

declare -A MODALITY_CONFIG
MODALITY_CONFIG["audio"]="ClothoAQA TUT2017" 
MODALITY_CONFIG["video"]="ActivityNet Perception MVBench"
MODALITY_CONFIG["audio_video"]="AVQA MusicAVQA AVSD"

for MODALITY in "audio" "video" "audio_video" ; do
    DATASETS=(${MODALITY_CONFIG[$MODALITY]}) 

    for DATASET in "${DATASETS[@]}"; do

        output_file=${OUTPUT_DIR}/${MODALITY}/${DATASET}/merge.jsonl
        rm ${output_file}

        if [ ! -f "$output_file" ]; then
            for IDX in $(seq 0 $((CHUNKS-1))); do
                # select the GPUs for the task
                gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
                TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python javisgpt/eval/infer_und.py \
                    --model_name_or_path ${CKPT_PATH} \
                    --beats-path ${BEATS_PATH} \
                    --all_projector_path ${ALL_PROJ_PATH} \
                    --lora_weight_path ${MODEL_PATH} \
                    --avsync_mode ${AVSYNC_MODE} \
                    --frames_upbound ${NUM_FRAMES} \
                    --frame_width ${FRAME_WIDTH} \
                    --frame_height ${FRAME_HEIGHT} \
                    --audio_sr ${AUDIO_SR} \
                    --max_audio_length_s ${MAX_AUDIO_LENGTH_S} \
                    --force_sample True \
                    --bf16 True \
                    --model_max_length ${MODEL_MAX_LENGTH} \
                    --batch-size 1 \
                    --num-workers 8 \
                    --dataset ${DATASET} \
                    --audio-folder ${EVAL_DATA_ROOT}/data \
                    --video-folder ${EVAL_DATA_ROOT}/data \
                    --image-folder ${EVAL_DATA_ROOT}/data \
                    --question-file ${EVAL_DATA_ROOT}/meta/${MODALITY}/${DATASET}.json \
                    --output-file ${OUTPUT_DIR}/${MODALITY}/${DATASET}/${CHUNKS}_${IDX}.jsonl \
                    --num-chunks $CHUNKS \
                    --chunk-idx $IDX  #>./run_logs/eval_${MODALITY}_${DATASET}_${CHUNKS}_${IDX}.log 2>./run_logs/eval_${MODALITY}_${DATASET}_${CHUNKS}_${IDX}.error &
            done

            wait

            # Clear out the output file if it exists.
            > "$output_file"

            #Loop through the indices and concatenate each file.
            for IDX in $(seq 0 $((CHUNKS-1))); do
                cat ${OUTPUT_DIR}/${MODALITY}/${DATASET}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
            done
        fi

    done
done

done
done