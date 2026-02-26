import argparse
import os.path as osp
import ast
import json
import math
import contextlib
import numpy as np
from copy import deepcopy
from typing import List


eval_data_cfg = {
    "audio": ["ClothoAQA", "TUT2017"],
    "video": ["ActivityNet", "Perception", "MVBench"],
    "audio_video": ["AVQA", "MusicAVQA", "AVSD"],
}


def calc_mc_accuracy(predicts: List[str], answers: List[str], **kwargs):
    acc_list = []
    for pred, label in zip(predicts, answers):
        if len(pred.strip()) == 0:
            acc_list.append(False)
            continue
        pred_option = pred.strip()[0]
        label_option = label.strip()[0]
        acc_list.append(pred_option == label_option)
    return np.mean(acc_list)


def calc_qa_accuracy(predicts: List[str], answers: List[str], **kwargs):
    acc_list = []
    for pred, label in zip(predicts, answers):
        pred = pred.strip().rstrip(' .').lower()
        label = label.strip().rstrip(' .').lower()
        acc_list.append(pred == label or label in pred or pred in label)
    return np.mean(acc_list)


def calc_oe_accuracy(predicts: List[str], answers: List[str], model, **kwargs):
    message = [
        {
            "role": "system",
            "content":
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
        },
        {
            "role": "user",
            "content":
                "Please evaluate the following video-based question-answer pair:\n\n"
                "Question: {question}\n"
                "Correct Answer: {answer}\n"
                "Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: "
                "{{'pred': 'yes', 'score': 4.8}}."
        }
    ]
    if isinstance(model, str):  # online API
        import asyncio
        from openai import AsyncOpenAI
        from tqdm.asyncio import tqdm
        client: AsyncOpenAI = kwargs['client']

        async def fetch_response(semaphore, messages, idx):
            async with semaphore:
                try:
                    resp = await client.chat.completions.create(
                        model=model,  # e.g., "gpt-4o-mini"
                        messages=messages,
                        **kwargs.get('sampling_params', {})
                    )
                    response = resp.choices[0].message.content
                except Exception as e:
                    response = str(e)
                return {'idx': idx, 'response': response}
        
        messages_list = []
        questions = kwargs.get('questions', [''] * len(answers))
        for question, pred, answer in zip(questions, predicts, answers):
            message_tmpl = deepcopy(message)
            message_tmpl[1]['content'] = message_tmpl[1]['content'].format(
                question=question, pred=pred, answer=answer
            )
            messages_list.append(message_tmpl)

        async def batch_process():
            semaphore = asyncio.Semaphore(10)  # MAX_CONCURRENT_REQUESTS
            tasks = [fetch_response(semaphore, msg, idx) for idx, msg in enumerate(messages_list)]
            results = await tqdm.gather(*tasks)
            results.sort(key=lambda x: x['idx'])
            responses = [res['response'] for res in results]
            return responses

        responses = asyncio.run(batch_process())

    else:  # local model
        from transformers import PreTrainedTokenizer
        tokenizer: PreTrainedTokenizer = model.get_tokenizer()
        template = tokenizer.decode(tokenizer.apply_chat_template(message)) + '<|im_start|>assistant\n'

        questions = kwargs.get('questions', [''] * len(answers))
        prompts = [template.format(question=question, pred=pred, answer=answer) \
                    for question, pred, answer in zip(questions, predicts, answers)]
        outputs = model.generate(prompts, sampling_params=kwargs['sampling_params'], use_tqdm=True)
        responses = [output.outputs[0].text for output in outputs]
    
    kwargs['eval_tmp'].update({'judge': responses})

    acc_list = []
    for response in responses:
        response_dict = ast.literal_eval(response)
        try:
            pred, score = response_dict['pred'], response_dict['score']
            correct = not (pred == 'no' and score < 3.0)
        except:
            print(f'failed to parse {response_dict}. set as 0.')
            correct = 0
        acc_list.append(correct)

    return np.mean(acc_list)


def build_judge_llm(jugde_model_name_or_path, api_key=""):
    if not jugde_model_name_or_path:
        return {}
    
    sampling_params = {
        "temperature": 0.01,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "max_tokens": 256,
        # "stop_token_ids"=[],
    }
    
    if 'OpenAI' in jugde_model_name_or_path:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        
        return {'client': client, 'sampling_params': sampling_params,
                'model': jugde_model_name_or_path.replace('OpenAI/', '')}

    else:
        import torch
        from vllm import LLM, SamplingParams

        model = LLM(
            model=jugde_model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            # gpu_memory_utilization=0.7
        )
        sampling_params = SamplingParams(**sampling_params)

        return {'model': model, 'sampling_params': sampling_params}


def run_eval(args):
    eval_kwargs = {'eval_tmp': {}}
    eval_kwargs.update(build_judge_llm(args.judge_model_name_or_path, args.api_key))

    eval_res_file = f'{args.res_dir}/eval_res.json'
    if osp.exists(eval_res_file):
        with open(eval_res_file, 'r') as f:
            res_dict = json.load(f)
    else:
        res_dict = {}

    s = ""
    modalities = list(eval_data_cfg.keys()) if args.modality == ['all'] else args.modality
    for modality in modalities:
        res_dict[modality] = res_dict.get(modality, {})
        datasets = eval_data_cfg[modality] if args.dataset == ['all'] else args.dataset
        s += f'{modality:12s}: '
        for dataset in datasets:
            if dataset in res_dict[modality]:
                print(f'### Already evaluated with {modality} with {dataset}. skip.')
                s += f'{dataset:12s} = {res_dict[modality][dataset]*100:.2f}%; '
                continue
            print(f'### Start evaluating {modality} with {dataset}')

            res_file = f'{args.res_dir}/{modality}/{dataset}/merge.jsonl'
            with open(res_file, 'r') as f:
                res_all = [json.loads(line) for line in f]
            predicts, answers = zip(*[[item['response'], item['answer']] for item in res_all])

            if dataset in ['MusicAVQA']:
                acc_func = calc_qa_accuracy
            elif dataset in ['ClothoAQA', 'AVSD', 'ActivityNet']:
                acc_func = calc_oe_accuracy
                eval_kwargs.update({'questions': [item['question'].strip() for item in res_all]})
            else:
                acc_func = calc_mc_accuracy
            
            accuracy = acc_func(predicts, answers, **eval_kwargs)
            res_dict[modality][dataset] = accuracy
            s += f'{dataset:12s} = {accuracy*100:.2f}%; '
            if (judge := eval_kwargs['eval_tmp'].pop('judge', None)) is not None:
                for item, acc in zip(res_all, judge):
                    item['judge'] = acc
                    del item['question']
                judge_res_file = res_file.replace('.jsonl', '_judge.jsonl')
                with open(judge_res_file, 'w+') as f:
                    f.writelines([json.dumps(item) + '\n' for item in res_all])
        s += '\n'

    print(s)
    with open(eval_res_file, 'w+') as f:
        json.dump(res_dict, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Configuration
    parser.add_argument('--res_dir', type=str, required=True)
    parser.add_argument("--modality", type=str, nargs='+', default=['all'])
    parser.add_argument("--dataset", type=str, nargs='+', default=['all'])
    parser.add_argument("--judge_model_name_or_path", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)