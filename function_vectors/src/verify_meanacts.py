#!/usr/bin/env python
"""
Equivalence check for the load-once driver.

Loads GPT-J ONCE, processes another task's mean-activations first (to exercise the
sequential in-process orchestration the driver uses), then recomputes GPT-J/antonym mean
head activations via the exact same preamble run_evaluation uses, and compares against the
cached smoke-test file with torch.allclose.

Exit 0 + "ALLCLOSE True" => the load-once path is computationally equivalent -> use driver.
Otherwise => fall back to per-task CLI invocations.
"""
import numpy as np
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset
from utils.eval_utils import n_shot_eval_no_intervention
from utils.extract_utils import get_mean_head_activations

SEED = 42
P = {"input": "Q:", "output": "A:", "instructions": ""}
S = {"input": "\n", "output": "\n\n", "instructions": ""}
CACHED = '../../results_repro/EleutherAI_gpt-j-6b/antonym/antonym_mean_head_activations.pt'


def compute_meanacts(task, model, tokenizer, model_config):
    # Mirror run_evaluation's preamble exactly (the parts that affect mean activations).
    set_seed(SEED)
    dataset = load_dataset(task, root_data_dir='../dataset_files', test_size=0.3, seed=SEED)
    set_seed(SEED + 42)
    fs_val = n_shot_eval_no_intervention(dataset=dataset, n_shots=10, model=model, model_config=model_config,
                                         tokenizer=tokenizer, compute_ppl=True, test_split='valid',
                                         prefixes=P, separators=S)
    filter_set_validation = np.where(np.array(fs_val['clean_rank_list']) == 0)[0]
    set_seed(SEED)
    ma = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer,
                                   n_icl_examples=10, N_TRIALS=100, prefixes=P, separators=S,
                                   filter_set=filter_set_validation)
    return ma


def main():
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer('EleutherAI/gpt-j-6b', device='cuda')

    # Process a DIFFERENT task first, in the same process, to exercise sequential orchestration.
    print("Processing 'capitalize' first (throwaway, to dirty in-process state) ...", flush=True)
    _ = compute_meanacts('capitalize', model, tokenizer, model_config)

    print("Recomputing 'antonym' mean activations via load-once path ...", flush=True)
    ma = compute_meanacts('antonym', model, tokenizer, model_config)

    cached = torch.load(CACHED)
    ma_c, cached_c = ma.float().cpu(), cached.float().cpu()
    match = torch.allclose(ma_c, cached_c)
    maxdiff = (ma_c - cached_c).abs().max().item()
    print(f"shape_new={tuple(ma.shape)} shape_cached={tuple(cached.shape)}")
    print(f"ALLCLOSE {match} MAXDIFF {maxdiff:.3e}")
    print("VERDICT " + ("PASS -> use load-once driver" if match else "FAIL -> fall back to per-task CLI"))


if __name__ == '__main__':
    main()
