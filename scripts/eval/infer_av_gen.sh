
USER_ROOT=${USER_ROOT:-$(dirname $(dirname "$PWD"))}  # ../../

PROJ_ROOT=${PROJ_ROOT:-"${USER_ROOT}/projects/JavisGPT"}
DATA_ROOT=${DATA_ROOT:-"${USER_ROOT}/datasets/JavisGPT"}
WEIGHT_ROOT=${WEIGHT_ROOT:-"${USER_ROOT}/weights"}
EVAL_DATA_PATH="${DATA_ROOT}/eval/JavisBench/JavisBench-mini.csv"
OUTPUT_DIR=${OUTPUT_DIR:-"results/javisgpt_v10/av_gen"}
GPU=${CUDA_VISIBLE_DEVICES:-"0"}

export BASE_ARCH="Qwen2_5_VL"
CKPT_PATH="${WEIGHT_ROOT}/pretrained/mllm/Qwen2.5-VL-7B-Instruct"
MODEL_MAX_LENGTH=131072
BEATS_PATH="${WEIGHT_ROOT}/pretrained/mllm/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

MODEL_PATH=${MODEL_PATH:-"${PROJ_ROOT}/runs/javisgpt_stage3_mm_insttune"}
ALL_PROJ_PATH="${MODEL_PATH}/mm_proj_all.bin"


JAV_VERSION=${JAV_VERSION:-"v1.0"}  # v0.1 or v1.0
if [ "$JAV_VERSION" = "v0.1" ]; then
    export AV_GEN_TOKEN_NUM=377 # JavisDiT
elif [ "$JAV_VERSION" = "v1.0" ]; then
    export AV_GEN_TOKEN_NUM=512 # JavisDiT++
else
    echo "Unknown JAV_VERSION: ${JAV_VERSION}"
fi
AVGEN_CFG_PATH="${PROJ_ROOT}/interface/config/javisdit_${JAV_VERSION}.py"


CUDA_VISIBLE_DEVICES=${GPU} python javisgpt/eval/infer_gen.py \
    --model_name_or_path ${CKPT_PATH} \
    --beats-path ${BEATS_PATH} \
    --all_projector_path ${ALL_PROJ_PATH} \
    --avgen_cfg_path ${AVGEN_CFG_PATH} \
    --lora_weight_path ${MODEL_PATH} \
    --dataset JavisBench \
    --question-file ${EVAL_DATA_PATH} \
    --bf16 True \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --output-dir ${OUTPUT_DIR}