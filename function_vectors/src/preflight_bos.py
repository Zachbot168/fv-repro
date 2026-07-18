#!/usr/bin/env python
"""
BOS / token-ID preflight for the modern tokenizers.

For each model, constructs ONE real 10-shot antonym prompt exactly as the eval path does
(n_shot_eval), tokenizes it exactly as sentence_eval does, and prints the leading token IDs
so we can confirm there is no double-BOS before committing 40h of compute.

Reminder of the pipeline's convention:
  prepend_bos_token = False if model_config['prepend_bos'] else True
So when the tokenizer auto-adds a BOS (prepend_bos=True, e.g. Llama-3), the prompt string does
NOT prepend one; when it does not (prepend_bos=False, e.g. Qwen), the prompt string prepends a
literal '<|endoftext|>'. Either way the expectation is exactly ONE leading BOS-like token.
"""
import sys
import numpy as np
import torch

from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt, get_token_meta_labels

P = {"input": "Q:", "output": "A:", "instructions": ""}
S = {"input": "\n", "output": "\n\n", "instructions": ""}
MODELS = sys.argv[1:] or ['meta-llama/Llama-3.1-8B', 'Qwen/Qwen2.5-7B']


def check(model_name):
    print("\n" + "=" * 78)
    print(f"MODEL: {model_name}")
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device='cuda')
    prepend_bos = False if model_config['prepend_bos'] else True
    print(f"  model_config['prepend_bos'] = {model_config['prepend_bos']}  ->  prompt prepend_bos_token = {prepend_bos}")
    print(f"  tokenizer.bos_token = {tokenizer.bos_token!r}  bos_token_id = {tokenizer.bos_token_id}")
    print(f"  tokenizer.add_bos_token attr = {getattr(tokenizer, 'add_bos_token', 'N/A')}")

    dataset = load_dataset('antonym', root_data_dir='../dataset_files', test_size=0.3, seed=42)
    word_pairs = dataset['train'][np.arange(10)]
    word_pairs_test = dataset['valid'][np.array([0])]
    prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test,
                                             prepend_bos_token=prepend_bos, shuffle_labels=False,
                                             prefixes=P, separators=S)
    sentence = create_prompt(prompt_data)

    # ---- eval path: exactly what sentence_eval does ----
    ids = tokenizer(sentence, return_tensors='pt').input_ids.squeeze().tolist()
    print(f"\n  [eval path]  prompt starts: {sentence[:40]!r}")
    print(f"  first 8 token ids : {ids[:8]}")
    print(f"  first 8 decoded   : {[tokenizer.decode([i]) for i in ids[:8]]}")

    bos_id = tokenizer.bos_token_id
    # Count leading tokens equal to any plausible BOS-like id.
    special_leading = []
    for i in ids:
        if i in (bos_id,) or tokenizer.decode([i]) in ('<|endoftext|>', '<|begin_of_text|>', '<s>'):
            special_leading.append(i)
        else:
            break
    n_leading = len(special_leading)
    print(f"  leading BOS-like tokens: {n_leading} ({special_leading})")

    # ---- mean-acts path: get_token_meta_labels tokenization ----
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer,
                                                        query=prompt_data['query_target']['input'],
                                                        prepend_bos=model_config['prepend_bos'])
    print(f"  [mean-acts path] n_token_labels={len(token_labels)}  first label={token_labels[0][2]!r} first tok={token_labels[0][1]!r}")

    verdict = "OK (single leading BOS)" if n_leading == 1 else (
        "OK (no explicit BOS, tokenizer adds none)" if n_leading == 0 and bos_id is None else
        f"!!! DOUBLE-BOS / anomaly: {n_leading} leading BOS-like tokens")
    print(f"  VERDICT: {verdict}")
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    for m in MODELS:
        check(m)
