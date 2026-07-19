#!/usr/bin/env python
"""
GPU probe for the Mistral verification (Stage 3, Part A4).

For each model, using the ANTONYM task's cached mean activations + indirect effects:
  1. rebuild the FV exactly as the eval did (compute_function_vector, top-10 heads)
  2. report ||FV|| and the residual-stream norm at the best injection layer
     (last token of a real zero-shot antonym prompt), i.e. the injection's relative scale
  3. decode the FV through the model's unembedding (ln_f/norm + lm_head) and report the
     top-15 vocab tokens -- does the FV point at task-relevant (antonym-ish) tokens?

Run from function_vectors/src:  python ../../analysis/fv_probe.py
Writes analysis/fv_probe_results.json and prints a human-readable log.
"""
import os, sys, json, gc
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'function_vectors', 'src'))
from baukit import TraceDict
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from utils.extract_utils import compute_function_vector

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
R = os.path.join(REPO, 'results_repro')

# (results tag, HF name, antonym best-effect layer from the sweep)
MODELS = [
    ('meta-llama_Llama-3.1-8B', 'meta-llama/Llama-3.1-8B', 13),
    ('Qwen_Qwen2.5-7B', 'Qwen/Qwen2.5-7B', 16),
    ('mistralai_Mistral-7B-v0.3', 'mistralai/Mistral-7B-v0.3', 14),
    ('EleutherAI_gpt-j-6b', 'EleutherAI/gpt-j-6b', 11),   # fp32, load last
]

def probe(tag, hf_name, best_layer, out):
    print(f"\n{'='*70}\nPROBE {hf_name} (layer {best_layer})", flush=True)
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(hf_name, device='cuda')

    ma = torch.load(f'{R}/{tag}/antonym/antonym_mean_head_activations.pt')
    ie = torch.load(f'{R}/{tag}/antonym/antonym_indirect_effect.pt')
    fv, top_heads = compute_function_vector(ma, ie, model, model_config=model_config, n_top_heads=10)

    # 2. norms: FV vs residual stream at the injection layer, on a real zero-shot prompt
    set_seed(42)
    dataset = load_dataset('antonym', root_data_dir=os.path.join(REPO, 'function_vectors', 'dataset_files'),
                           test_size=0.3, seed=42)
    word_pairs = {'input': [], 'output': []}  # zero-shot: no ICL examples (mirrors n_shot_eval n_shots=0)
    q = dataset['test'][0]
    prepend_bos = False if model_config['prepend_bos'] else True
    pd_ = word_pairs_to_prompt_data(word_pairs, query_target_pair={'input': q['input'], 'output': q['output']},
                                    prepend_bos_token=prepend_bos)
    sentence = create_prompt(pd_)
    inputs = tokenizer(sentence, return_tensors='pt').to('cuda')
    layer_name = model_config['layer_hook_names'][best_layer]
    with TraceDict(model, layers=[layer_name]) as td:
        model(**inputs)
    h = td[layer_name].output
    h = h[0] if isinstance(h, tuple) else h
    resid_norm = h[0, -1].float().norm().item()
    fv_norm = fv.float().norm().item()

    # 3. decode FV to vocab through the model's own unembedding
    if 'gpt-j' in hf_name:
        dec = torch.nn.Sequential(model.transformer.ln_f, model.lm_head)
    else:
        dec = torch.nn.Sequential(model.model.norm, model.lm_head)
    logits = dec(fv.reshape(1, 1, -1).to(model.dtype).cuda()).float().squeeze()
    topv, topi = torch.topk(torch.softmax(logits, dim=-1), 15)
    top_tokens = [(tokenizer.decode([i]), round(v.item(), 4)) for i, v in zip(topi, topv)]

    rec = {'model': hf_name, 'layer': best_layer,
           'fv_norm': round(fv_norm, 2), 'resid_norm_lasttok': round(resid_norm, 2),
           'ratio': round(fv_norm / resid_norm, 3),
           'top_heads_LH_score': [(int(L), int(H), s) for L, H, s in top_heads[:10]],
           'fv_top_tokens': top_tokens}
    out[hf_name] = rec
    print(json.dumps(rec, indent=1), flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()

def main():
    out = {}
    for tag, hf, L in MODELS:
        try:
            probe(tag, hf, L, out)
        except Exception as e:
            print(f"PROBE_ERROR {hf}: {e!r}", flush=True)
            out[hf] = {'error': repr(e)}
        json.dump(out, open(os.path.join(REPO, 'analysis', 'fv_probe_results.json'), 'w'), indent=2)
    print("\nPROBE_ALL_DONE", flush=True)

if __name__ == '__main__':
    main()
