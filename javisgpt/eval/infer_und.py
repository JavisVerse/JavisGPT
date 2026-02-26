import os
import os.path as osp
import re
import sys
import json
import math
import math
import torch
import argparse
import warnings
from tqdm import tqdm

from javisgpt.train.train_audio_video import get_model_and_tokenizer
from javisgpt.utils import disable_torch_init, set_random_seed
from javisgpt.model.qwen2_vl import Qwen2VLImageProcessor
from javisgpt.eval.dataset import *

set_random_seed(42)

def run_inference(args):
    disable_torch_init()

    # Initialize the model
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model, tokenizer = get_model_and_tokenizer(args, args, maybe_merge_lora=True, is_training=False)
    model = model.to(device='cuda', dtype=dtype)
    model.eval()

    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_name_or_path)
    setattr(args, 'image_processor', image_processor)
    
    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    if osp.exists(args.output_file):  # auto-resume
        with open(args.output_file, 'r') as f:
            content = f.read().splitlines()
        last_question_id_tmpl = json.loads(content[-1].strip())['question_id']
        valid_questions, valid_flag = [], False
        for item in gt_questions:
            if item['id'] in last_question_id_tmpl:
                valid_flag = True
                continue
            if valid_flag:
                valid_questions.append(item)
        gt_questions = valid_questions

    if args.dataset in ['Perception', 'AVSD'] and args.batch_size > 1:
        warnings.warn(f'Turn batch size {args.batch_size} back to 1 for multi-turn eval datset {args.dataset}')
        args.batch_size = 1

    dataset, dataloader = build_dataloader(
        args.dataset, tokenizer, gt_questions, data_args=args,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    ans_file = open(args.output_file, "a")

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids = batch.pop('input_ids')
        question_ids, questions, answers, metas = \
            batch.pop("question_id"), batch.pop("question"), batch.pop("answer"), batch.pop("meta", [{}] * len(input_ids))
        with torch.inference_mode():
            generated_ids = model.generate(
                inputs=input_ids,
                do_sample=True, temperature=0.01, top_p=0.1, num_beams=1, max_new_tokens=256,
                **batch
            )
            output_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        
        for qid, qus, ans, res, meta in zip(question_ids, questions, answers, output_texts, metas):
            res_item = {'question_id': qid, 'question': qus, 'response': res, 'answer': ans, 'meta': meta}
            ans_file.write(json.dumps(res_item) + "\n")
        ans_file.flush()
            
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Configuration
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=False)
    parser.add_argument('--audio-folder', help='Directory containing audio files.', required=False)
    parser.add_argument('--image-folder', help='Directory containing image files.', required=False)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=False)
    parser.add_argument('--output-file', help='Directory to save the model results in jsonl.', required=True)
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
    parser.add_argument("--force_sample", type=bool, default=True)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--max_audio_length_s", type=float, default=None)
    args = parser.parse_args()

    run_inference(args)