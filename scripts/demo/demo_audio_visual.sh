
USER_ROOT=${USER_ROOT:-$(dirname $(dirname "$PWD"))}  # ../../

PROJ_ROOT="${USER_ROOT}/projects/JavisGPT"
WEIGHT_ROOT="${USER_ROOT}/weights"

export BASE_ARCH="Qwen2_5_VL"
MODEL_PATH="${WEIGHT_ROOT}/pretrained/mllm/Qwen2.5-VL-7B-Instruct"
BEATS_PATH="${WEIGHT_ROOT}/pretrained/mllm/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

JAV_VERSION=${JAV_VERSION:-"v1.0"}  # v0.1 or v1.0
if [ "$JAV_VERSION" = "v0.1" ]; then
    export AV_GEN_TOKEN_NUM=377 # JavisDiT
    AVGEN_CFG_PATH="${PROJ_ROOT}/interface/config/javisdit_v0.1.py"
elif [ "$JAV_VERSION" = "v1.0" ]; then
    export AV_GEN_TOKEN_NUM=512 # JavisDiT++
    AVGEN_CFG_PATH="${PROJ_ROOT}/interface/config/javisdit_v1.0.py"
else
    echo "Unknown JAV_VERSION: ${JAV_VERSION}"
fi

GPU=${GPU:-"0"}
CKPT_DIR=${CKPT_DIR:-"${WEIGHT_ROOT}/JavisVerse/JavisGPT-v0.1-7B-Instruct"}
IMAGE_PATH=${IMAGE_PATH:-""}
VIDEO_PATH=${VIDEO_PATH:-""}
AUDIO_PATH=${AUDIO_PATH:-""}
USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-"False"}
PROMPT=${PROMPT:-"Describe the input content in detail."}
AV_GENERATE=${AV_GENERATE:-"False"}
SAVE_PREFIX=${SAVE_PREFIX:-"./results/avgen/demo"}

AVUND_ARGS=""
if [ -n "$IMAGE_PATH" ]; then
    AVUND_ARGS+="--image_path ${IMAGE_PATH} "
fi
if [ -n "$VIDEO_PATH" ]; then
    AVUND_ARGS+="--video_path ${VIDEO_PATH} "
fi
if [ -n "$AUDIO_PATH" ]; then
    AVUND_ARGS+="--audio_path ${AUDIO_PATH} "
fi

AVGEN_ARGS=""
if [ "$AV_GENERATE" = "True" ]; then
    AVGEN_ARGS+="--avgen_cfg_path ${AVGEN_CFG_PATH} --av_generate True --prefill_avgen_token False --save_path_prefix ${SAVE_PREFIX} "
fi

CUDA_VISIBLE_DEVICES=${GPU} python tools/audio_visual_demo.py \
    --model_name_or_path "${MODEL_PATH}" \
    --beats_path "${BEATS_PATH}" \
    --lora_weight_path "${CKPT_DIR}" \
    --all_projector_path "${CKPT_DIR}/mm_proj_all.bin" \
    --bf16 True \
    ${AVUND_ARGS} ${AVGEN_ARGS} \
    --prompt "${PROMPT}"
