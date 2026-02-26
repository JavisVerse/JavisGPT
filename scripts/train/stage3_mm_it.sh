export TOKENIZERS_PARALLELISM=false
export ACCELERATE_BYPASS_DEVICE_MAP=true
export CUDA_LAUNCH_BLOCKING=1

USER_ROOT=${USER_ROOT:-$(dirname $(dirname "$PWD"))}  # ../../

PROJ_ROOT=${PROJ_ROOT:-"${USER_ROOT}/projects/JavisGPT"}
DATA_ROOT=${DATA_ROOT:-"${USER_ROOT}/datasets/JavisGPT"}
WEIGHT_ROOT=${WEIGHT_ROOT:-"${USER_ROOT}/weights"}

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}

IMAGE_FOLDER="${DATA_ROOT}/train/JavisInst-Omni"
VIDEO_FOLDER="${DATA_ROOT}/train/JavisInst-Omni"
AUDIO_FOLDER="${DATA_ROOT}/train/JavisInst-Omni"

DATASET_PATH="${DATA_ROOT}/train/JavisInst-Omni/stage3_mm_it.json"

JAV_VERSION=${JAV_VERSION:-"v1.0"}  # v0.1 or v1.0
if [ "$JAV_VERSION" = "v0.1" ]; then
    export AV_GEN_TOKEN_NUM=377 # JavisDiT
elif [ "$JAV_VERSION" = "v1.0" ]; then
    export AV_GEN_TOKEN_NUM=512 # JavisDiT++
else
    echo "Unknown JAV_VERSION: ${JAV_VERSION}"
fi

NUM_FRAMES=16
FRAME_WIDTH=588  # 448(4:3) or 588(16:9)
FRAME_HEIGHT=336
AUDIO_SR=16000
MAX_AUDIO_LENGTH_S=30
AVSYNC_MODE="merge"
AVGEN_CFG_PATH="${PROJ_ROOT}/interface/config/javisdit_${JAV_VERSION}.py"

NUM_EPOCHS=1
BATCH_SIZE=1
NUM_WORKERS=8
LEARNING_RATE=1e-4
LORA_ENABLE=True

RUN_NAME="javisgpt_stage3_mm_insttune"
echo "RUN_NAME: ${RUN_NAME}"

OUTPUT_DIR="${PROJ_ROOT}/runs/${RUN_NAME}"
LOG_DIR="${OUTPUT_DIR}/log"
mkdir -p ${LOG_DIR}

export BASE_ARCH="Qwen2_5_VL"
CKPT_PATH="${WEIGHT_ROOT}/pretrained/mllm/Qwen2.5-VL-7B-Instruct"
MODEL_MAX_LENGTH=131072
ALL_PROJ_PATH="${PROJ_ROOT}/runs/javisgpt_stage2_av_finetune/mm_proj_all.bin"
LORA_WEIGHT_PATH="${PROJ_ROOT}/runs/javisgpt_stage2_av_finetune"
BEATS_PATH="${WEIGHT_ROOT}/pretrained/mllm/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"


deepspeed --include="localhost:${GPUS}" --master_port 29620 \
    javisgpt/train/train_audio_video.py \
    --lora_enable ${LORA_ENABLE} --lora_r 128 --lora_alpha 256 \
    --deepspeed ${PROJ_ROOT}/scripts/zero2.json \
    --calc_dummy_loss True \
    --model_name_or_path ${CKPT_PATH} \
    --beats_path ${BEATS_PATH} \
    --all_projector_path ${ALL_PROJ_PATH} \
    --avgen_cfg_path ${AVGEN_CFG_PATH} \
    --lora_weight_path ${LORA_WEIGHT_PATH} \
    --data_path ${DATASET_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --audio_folder ${AUDIO_FOLDER} \
    --video_folder ${VIDEO_FOLDER} \
    --avsync_mode ${AVSYNC_MODE} \
    --frames_upbound ${NUM_FRAMES} \
    --frame_width ${FRAME_WIDTH} \
    --frame_height ${FRAME_HEIGHT} \
    --audio_sr ${AUDIO_SR} \
    --max_audio_length_s ${MAX_AUDIO_LENGTH_S} \
    --training_stage "mm_insttune" \
    --calc_dummy_loss True \
    --force_sample True \
    --bf16 True \
    --fp16 False \
    --run_name $RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${NUM_WORKERS} \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True >${LOG_DIR}/${RUN_NAME}_output.txt 2>${LOG_DIR}/${RUN_NAME}_error.txt &
