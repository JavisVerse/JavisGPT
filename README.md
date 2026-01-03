## <div align="center"> JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</div>

<div align="center">

[[`HomePage`](https://javisverse.github.io/JavisGPT-page/)] 
[[`HF Paper`](https://huggingface.co/papers/2512.22905)] 
[[`ArXiv Paper`](https://arxiv.org/abs/2512.22905)] 
[[`Model`](https://huggingface.co/collections/JavisVerse/javisgpt)] 
[[`Dataset`](https://huggingface.co/collections/JavisVerse/javisgpt)] 

</div>


## TL;DR

We introduce **`JavisGPT`**, a multimodal LLM that can understand audiovisual inputs and simultaneously generate synchronized sounding videos in a unified model. 
We also curate the **`JavisInst-Omni`** dataset to facilitate instruction-tuning for comprehension and generation on sounding videos.


![framework](./assets/image/framework.jpg)


## 📰 News

- **[2026.1.1]** 🚀 We release the full training and evaluation scripts to support future research in the community. Have fun with them!
- **[2025.12.30]** 🚀 We release the training dataset of [JavisInst-Omni](https://huggingface.co/datasets/JavisVerse/JavisInst-Omni) to support multimodal instruction tuning on sounding video comprehension and generation tasks, as well as [MM-PreTrain](https://huggingface.co/datasets/JavisVerse/MM-PreTrain) and [AV-FineTune](https://huggingface.co/datasets/JavisVerse/AV-FineTune) datasets to enable preliminary multimodal alignment for LLMs. The [JavisUnd-Eval](https://huggingface.co/datasets/JavisVerse/JavisUnd-Eval) dataset is also released to set a standard for audio-video understanding evaluation for MLLMs.
- **[2025.12.26]** 🔥 We release the code of [JavisGPT](https://arxiv.org/abs/2512.22905), with the preview [JavisGPT-v0.1-7B-Instruct](https://huggingface.co/JavisVerse/JavisGPT-v0.1-7B-Instruct) checkpoint at huggingface. Feel free to play with it!

### 👉 TODO 
- [ ] Derive a more powerful JavisGPT model.


## Code

We assume the data structure as:

```bash
/path/to/user/root
|-- projects
|   |   |-- JavisDiT  # downstream JAV-DiT
|   |   └-- JavisGPT  # workspace of this project
|-- weights
|   |-- pretrained
|   |   |-- dit   # pretrained weights for JavisDiT
|   |   |   |-- OpenSora-VAE-v1.2
|   |   |   |-- audioldm2
|   |   |   |-- t5-v1_1-xxl
|   |   |   |-- imagebind_huge.pth
|   |   |-- mllm  # pretrained weights for JavisGPT
|   |   |   |-- BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
|   |   |   └-- Qwen2.5-VL-7B-Instruct
|   |-- JavisVerse
|   |   |-- JavisDiT-v0.1-prior
|   |   |-- JavisDiT-v0.1-jav-240p4s
|   |   └-- JavisGPT-v0.1-7B-Instruct
|-- datasets
|   |-- JavisGPT
|   |   |-- train
|   |   |   |-- MM-PreTrain
|   |   |   |-- AV-FineTune
|   |   |   └-- JavisInst-Omni
|   |   |-- eval
|   |   |   |-- JavisUnd-Eval
|   |   |   └-- JavisBench
```


### Installation

Install the necessary packages:

```bash
conda create -n javisgpt python=3.10 -y
conda activate javisgpt
pip install --upgrade pip  # Enable PEP 660 support.
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -v -e ".[train]"
cp assets/src/dynamic_modules_utils.py /path/to/python3.10/site-packages/diffusers/utils/
conda install "ffmpeg<7" -c conda-forge -y  # install ffpmeg
```

Install [JavisDiT](https://github.com/JavisVerse/JavisDiT.git) dependencies:

```bash
cd ..
git clone https://github.com/JavisVerse/JavisDiT.git
cd JavisDiT
pip install -v -e . --no-deps
cd ../JavisGPT

# # make soft links if necessary
# ln -s ../JavisDiT/javisdit javisdit
```

### Inference

#### 1. Prepare Pretrained Weights

First, download [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt](https://github.com/microsoft/unilm/tree/master/beats) from [here](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), and put (or link) them into `../../weights/pretrained/mllm`.

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ../../weights/pretrained/mllm/Qwen2.5-VL-7B-Instruct
```

Then, download our [JavisGPT-v0.1-7B-Instruct](https://huggingface.co/JavisVerse/JavisGPT-v0.1-7B-Instruct) and put it into `../../weights/JavisVerse`.

```bash
hf download JavisVerse/JavisGPT-v0.1-7B-Instruct --local-dir ../../weights/JavisVerse/JavisGPT-v0.1-7B-Instruct
```

Finally, download necessary checkpoints of the downstream JAVG model ([JavisDiT](https://github.com/JavisVerse/JavisDiT.git)) and put them into `../../weights/pretrained/dit` or `../../weights/JavisVerse`, according to path definition in `./interface/config/*.py` coordinately.

```bash
hf download hpcai-tech/OpenSora-VAE-v1.2 --local-dir ../../weights/pretrained/dit/OpenSora-VAE-v1.2
hf download cvssp/audioldm2 --local-dir ../../weights/pretrained/dit/audioldm2
hf download DeepFloyd/t5-v1_1-xxl --local-dir ../../weights/pretrained/dit/t5-v1_1-xxl
hf download JavisVerse/JavisDiT-v0.1-prior --local-dir ../../weights/JavisVerse/JavisDiT-v0.1-prior
hf download JavisVerse/JavisDiT-v0.1-jav-240p4s --local-dir ../../weights/JavisVerse/JavisDiT-v0.1-jav-240p4s
```

#### 2. Run Target Inference

- **Standalone Audio/Visual Comprehension**

Use the following commands to evaluate the preserved single-modality understanding capability.

For audio comprehension:

```bash
AUDIO_PATH="assets/demos/audio/Creaking_pier.wav"
PROMPT="Is the sound caused by pressure from/against wood?"

AUDIO_PATH=${AUDIO_PATH} PROMPT=${PROMPT} bash scripts/demo/demo_audio_visual.sh
```

For video comprehension:

```bash
VIDEO_PATH="assets/demos/video/ZS9XR.mp4"
PROMPT="What happened after the person took the box? A. Ate the medicine. B. Tidied up the blanket. C. Put down the cup/glass/bottle. D. Open the computer."

VIDEO_PATH=${VIDEO_PATH} PROMPT=${PROMPT} bash scripts/demo/demo_audio_visual.sh
```

- **Joint Audio-Video Comprehension**

Use the following command to evaluate the joint audio-video comprehension capability.


```bash
VIDEO_PATH="assets/demos/audio_video/00002617.mp4"
PROMPT="How many instruments in the room did not sound from beginning to end? Answer the question using a single word."
USE_AUDIO_IN_VIDEO=True

VIDEO_PATH=${VIDEO_PATH} PROMPT=${PROMPT} USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO} bash scripts/demo/demo_audio_visual.sh
```


- **Joint Audio-Video Generation**

Use the following command to evaluate the sounding video generation capability.

```bash
PROMPT="Build a video, ensuring the content is echoed by complementary scenes: A beautiful waterfall cascades down a steep cliff into a clear pool below. Sunlight filters through the surrounding trees, creating shimmering reflections on the falling water. The scene is calm and natural, with continuous flowing water and gentle mist rising from the base. The sound consists of steady rushing water, soft splashes, and faint ambient forest noise."
AV_GENERATE=True
SAVE_PREFIX="./results/avgen/demo"

AV_GENERATE=${AV_GENERATE} PROMPT=${PROMPT} SAVE_PREFIX=${SAVE_PREFIX} bash scripts/demo/demo_audio_visual.sh
```

The generated sample will be saved at `${SAVE_PREFIX}.mp4`, e.g., `./results/avgen/demo.mp4`.

### Training


#### 1. Prepare Training Datasets

Download the corresponding [MM-PreTrain](https://huggingface.co/datasets/JavisVerse/MM-PreTrain), [AV-FineTune](https://huggingface.co/datasets/JavisVerse/AV-FineTune), and [JavisInst-Omni](https://huggingface.co/datasets/JavisVerse/JavisInst-Omni) from [HuggingFace](https://huggingface.co/collections/JavisVerse/javisgpt), and put (or link) them into `../../datasets/JavisGPT/train/` coordinately.

```bash
hf download --repo-type dataset JavisVerse/MM-PreTrain --local-dir ../../datasets/JavisGPT/train/MM-PreTrain
hf download --repo-type dataset JavisVerse/AV-FineTune --local-dir ../../datasets/JavisGPT/train/AV-FineTune
hf download --repo-type dataset JavisVerse/JavisInst-Omni --local-dir ../../datasets/JavisGPT/train/JavisInst-Omni
```

Then, go to the downloaded datasets and run the `unzip.py` to extract source audio/video data:

```bash
cd ../../datasets/JavisGPT/train/MM-PreTrain # MM-PreTrain/AV-FineTune/JavisInst-Omni
python unzip.py --purge
```

However, we cannot release the source data of [TAVGBench](https://arxiv.org/abs/2404.14381) due to policy issues. Instead, the video_ids (formatted with `{youtube_id}_{start_time}_{end_time}`) are provided in [`video_ids.txt`](video_ids.txt) at [AV-FineTune](https://huggingface.co/datasets/JavisVerse/AV-FineTune) and [JavisInst-Omni](https://huggingface.co/datasets/JavisVerse/JavisInst-Omni), and users can refer to [TAVGBench](https://github.com/OpenNLPLab/TAVGBench) to download raw videos.


#### 2. Conduct Progressive Training Procedure

- **Stage-I Multi-Modal Pre-Training**

This stage aims to enquip the backbone Qwen2.5-VL-Instruct model with preliminary audio comprehension and audio-video generation capabilities, includes two distinct tasks: (1) _audio comprehension pretraining_ and (2) _audio–video generation pretraining_. We adopt separate training for data efficiency, since the two tasks do not share gradient interactions during optimization.

For audio comprehension pretraining, run the following command, and the trained `audio_proj.bin` will be saved at `./runs/javisgpt_stage1_mm_pretrain_audio_align/`:

```bash
bash ./scripts/train/train_audio_align.sh
```

For audio-video generation pretraining, run the following command, and the trained `avgen_proj.bin` will be saved at `./runs/javisgpt_stage1_mm_pretrain_avgen_align/`:

```bash
bash ./scripts/train/train_audio_video_gen_align.sh
```

- **Stage-II Audio-Visual Fine-Tuning**

This stage aims to enhance the understanding and generation of sounding videos, which can be integrated into a single task. Run the following command, and the trained `mm_proj_all.bin` with LoRA weights will be saved at `./runs/javisgpt_stage2_av_finetune/`:

```bash
bash ./scripts/train/stage2_av_ft.sh
```

- **Stage-III Multi-Modal Insturction-Tuning**

This stage aims to elicit the multimodal instruction-following ability for audio/video/audio-video comprehension and joint audio-video generation tasks.
Run the following command, and the trained `mm_proj_all.bin` with LoRA weights will be saved at `./runs/javisgpt_stage3_mm_insttune/`:

```bash
bash ./scripts/train/stage3_mm_it.sh
```

Only the checkpoints trained in this stage are utilized to build the final JavisGPT model.

### Evaluation

#### 1. Prepare Evaluation Datasets

- **Audio/Video/Audio-Video Comprehension**

Download and extract the preprocessed [JavisUnd-Eval](https://huggingface.co/datasets/JavisVerse/JavisUnd-Eval) dataset (including 8 widely-used subsets) to `../../datasets/JavisGPT/eval/`:

```bash
hf download --repo-type dataset JavisVerse/JavisUnd-Eval --local-dir ../../datasets/JavisGPT/eval/JavisUnd-Eval

cd ../../datasets/JavisGPT/eval/JavisUnd-Eval
python unzip.py --purge
cd -
```

- **Joint Audio-Video Generation**

Download the [JavisBench](https://huggingface.co/datasets/JavisVerse/JavisBench) dataset to `../../datasets/JavisGPT/eval/`:

```bash
hf download --repo-type dataset JavisVerse/JavisBench --local-dir ../../datasets/JavisGPT/eval/JavisBench
```

#### 2. Run Target Inference

- **Audio/Video/Audio-Video Comprehension**

Run the following script to automatically collect the responses from audio (`ClothoAQA`, `TUT2017`), video (`ActivityNet`, `Perception`, `MVBench`), and audio-video (`AVQA`, `MusicAVQA`, `AVSD`) datasets:

```bash
bash ./scripts/eval/infer_av_und.sh
```

Model's responses will be saved at `./results/av_und/`.

- **Joint Audio-Video Generation**

Run the following script to automatically collect the joint audio-video generation results from `JavisBench-mini`:

```bash
bash ./scripts/eval/infer_av_gen.sh
```

Model's responses will be saved at `./results/av_gen/`.


#### 3. Run Target Evaluation


- **Audio/Video/Audio-Video Comprehension**

If you use local models as the LLM judge, such as [Qwen series](https://huggingface.co/collections/Qwen/qwen25), install the `vllm` dependency and run the following commands:

```bash
pip install vllm==0.7.3

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

judge_model_name_or_path="Qwen/Qwen2.5-72B-Instruct"
res_dir="./results/av_und/"

python javisgpt/eval/eval_und.py \
    --res_dir "${res_dir}" \
    --judge_model_name_or_path "${judge_model_name_or_path}"
```

Otherwise, if you use OpenAI API as the LLM judge, run:

```bash
api_key="YOU_API_KEY"
judge_model_name_or_path="OpenAI/gpt-4o-mini"
res_dir="./results/av_und/"

python javisgpt/eval/eval_und.py \
    --res_dir "${res_dir}" \
    --judge_model_name_or_path "${judge_model_name_or_path}" \
    --api_key "${api_key}"
```

The evaluation results will be saved at `${api_key}/eval_res.json`, e.g., `./results/av_und/eval_res.json`

- **Joint Audio-Video Generation**

We reuse the scripts proposed in [JavisDiT](https://github.com/JavisVerse/JavisDiT.git) to evaluate the quality, consistency, and synchrony of generated sounding videos:

```bash
cd ../JavisDiT

INFER_DATA_DIR="../JavisGPT/results/av_gen"
EVAL_DATA_ROOT="../../datasets/JavisGPT/eval/JavisBench"
RESULTS_DIR="./evaluation_results/JavisGPT"

MAX_FRAMES=16
IMAGE_SIZE=224
MAX_AUDIO_LEN_S=4.0

# Params to calculate JavisScore
WINDOW_SIZE_S=2.0
WINDOW_OVERLAP_S=1.5

METRICS="all" 

DATASET="JavisBench-mini"
INPUT_FILE="${EVAL_DATA_ROOT}/${DATASET}.csv"
FVD_AVCACHE_PATH="${EVAL_DATA_ROOT}/cache/fvd_fad/${DATASET}-vanilla-max4s.pt"

python -m eval.javisbench.main \
  --input_file "${INPUT_FILE}" \
  --infer_data_dir "${INFER_DATA_DIR}" \
  --output_file "${RESULTS_DIR}/${DATASET}.json" \
  --max_frames ${MAX_FRAMES} \
  --image_size ${IMAGE_SIZE} \
  --max_audio_len_s ${MAX_AUDIO_LEN_S} \
  --window_size_s ${WINDOW_SIZE_S} \
  --window_overlap_s ${WINDOW_OVERLAP_S} \
  --fvd_avcache_path ${FVD_AVCACHE_PATH} \
  --metrics ${METRICS}
```

The evaluation results will be saved at `${JavisDiT_ROOT}/evaluation_results/JavisGPT/JavisBench-mini.json`.

## Citation

If you find JavisGPT is useful and use it in your project, please kindly cite:
```
@inproceedings{liu2025javisgpt,
    title={JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation},
    author={Kai Liu and Jungang Li and Yuchong Sun and Shengqiong Wu and jianzhang gao and Daoan Zhang and Wei Zhang and Sheng Jin and Sicheng Yu and Geng Zhan and Jiayi Ji and Fan Zhou and Liang Zheng and Shuicheng YAN and Hao Fei and Tat-Seng Chua},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```

