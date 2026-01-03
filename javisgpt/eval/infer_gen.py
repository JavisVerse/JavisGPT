import os
import os.path as osp
import math
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from javisgpt.constants import *
from javisgpt.train.train_audio_video import get_model_and_tokenizer
from javisgpt.utils import disable_torch_init, set_random_seed
from javisgpt.eval.dataset import build_dataloader

set_random_seed(42)


@torch.no_grad()
def run_inference(args):
    disable_torch_init()

    # Initialize the model
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model, tokenizer = get_model_and_tokenizer(args, args, maybe_merge_lora=True, is_training=False)
    model = model.to(device='cuda', dtype=dtype)
    model.eval()

    assert args.batch_size == 1
    dataset, dataloader = build_dataloader(
        args.dataset, tokenizer, args.question_file, data_args=args, 
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    eos_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN, DEFAULT_AUDIO_VIDEO_START_TOKEN])

    for bi, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids, raw_caption = batch.pop('input_ids'), batch.pop('raw_caption')
        if all(osp.exists(f"{output_dir}/{bi*args.batch_size+di:04d}.mp4") for di in range(len(input_ids))):
            continue
        question_ids = batch.pop("question_id")
        with torch.inference_mode():
            input_ids_bak = input_ids.clone()
            outputs = model.generate(
                inputs=input_ids,
                do_sample=True, temperature=0.01, top_p=0.1, num_beams=1,
                eos_token_id=eos_token_id, max_new_tokens=256, 
                return_dict_in_generate=True, output_hidden_states=True,
                **batch
            )
            # generated_ids = outputs.sequences
            # output_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:])

            hidden_states = outputs.hidden_states[0][-1]
            avgen_embeds = model.parse_avgen_embed(hidden_states, input_ids_bak, GEN_AUDIO_VIDEO_TOKEN_INDEX)
            assert avgen_embeds.shape[0] == args.batch_size, f"{avgen_embeds.shape} vs {args.batch_size}"

            audios, videos = model.generate_audio_video_direct(avgen_embeds)
            for di, (audio, video) in enumerate(zip(audios, videos)):
                model.av_generator.save(audio, video, save_path=f"{output_dir}/sample_{bi*args.batch_size+di:04d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Configuration
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-dir', help='Directory to save the generated audio-video results.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=1)
    # Model arguments
    parser.add_argument('--model_name_or_path', help='', required=True)
    parser.add_argument('--beats-path', help='', required=True)
    parser.add_argument('--audio_projector_path', help='', required=False)
    parser.add_argument("--avsync_mode", type=str, default="merge")
    parser.add_argument("--avsync_projector_path", type=str)
    parser.add_argument("--avgen_cfg_path", type=str)
    parser.add_argument("--avgen_projector_path", type=str)
    parser.add_argument('--all_projector_path', type=str, default=None)
    parser.add_argument('--lora_weight_path', type=str, default='')
    parser.add_argument("--tokenizer_path", type=str, default=None, help="deprecated")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    # Process arguments
    parser.add_argument("--video_fps", type=int, default=2)
    parser.add_argument("--frames_upbound", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--frame_width", type=int, default=588)
    parser.add_argument("--frame_height", type=int, default=336)
    parser.add_argument("--force_sample", type=bool, default=False)  # False for test
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--max_audio_length_s", type=float, default=None)
    args = parser.parse_args()

    run_inference(args)