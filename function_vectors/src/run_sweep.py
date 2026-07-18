#!/usr/bin/env python
"""
Load-once driver for the FV generalizability sweep.

Orchestration only: for each model it loads the weights ONCE and then calls
evaluate_function_vector.run_evaluation() for every task, reusing the loaded model.
The per-task computation is identical to invoking evaluate_function_vector.py from the
CLI (verified via mean-activation allclose against the GPT-J/antonym smoke-test file).

Layout (canonical, single-nested):
  results_repro/<model_tag>/<task>/              main run (sweep over all layers, top-10 heads)
  results_repro/<model_tag>/k<K>/<task>/         k-ablation at the main run's best layer

Resume-safe: a task whose completion marker exists is skipped, so a restart after a crash
or disconnect continues without redoing finished work.
"""
import os, json, time, gc, shutil
from types import SimpleNamespace
import numpy as np
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: F401 (set_seed used inside run_evaluation)
from evaluate_function_vector import run_evaluation

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, '..', '..'))
RESULTS_ROOT = os.environ.get('RESULTS_ROOT', os.path.join(_REPO_ROOT, 'results_repro'))
ROOT_DATA_DIR = os.path.join(_SRC_DIR, '..', 'dataset_files')
SEED = 42
PROGRESS_LOG = os.path.join(RESULTS_ROOT, 'sweep_progress.log')
CONTROL_FILE = os.path.join(RESULTS_ROOT, 'sweep_control.json')

# Order per instructions: modern models first, GPT-J (slow fp32) last.
MODELS = [
    'meta-llama/Llama-3.1-8B',
    'Qwen/Qwen2.5-7B',
    'mistralai/Mistral-7B-v0.3',
    'EleutherAI/gpt-j-6b',
]
TASKS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']
KABLATION_TASKS = ['antonym', 'country-capital']
KABLATION_KS = [5, 20, 50]
DEFAULT_TOP_HEADS = 10

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def log(msg):
    line = f"[{int(time.time())}] {msg}"
    print(line, flush=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(PROGRESS_LOG, 'a') as f:
        f.write(line + "\n")


def tag_of(model_name):
    return model_name.replace('/', '_')


def skip_shuffled_now():
    try:
        with open(CONTROL_FILE) as f:
            return bool(json.load(f).get('skip_shuffled', False))
    except Exception:
        return False


def make_args(dataset_name, model_name, save_path_root, ie_path_root, edit_layer, n_top_heads, compute_baseline):
    return SimpleNamespace(
        dataset_name=dataset_name, model_name=model_name, root_data_dir=ROOT_DATA_DIR,
        save_path_root=save_path_root, ie_path_root=ie_path_root, seed=SEED,
        device='cuda', mean_activations_path=None, indirect_effect_path=None,
        n_top_heads=n_top_heads, edit_layer=edit_layer, test_split=0.3, n_shots=10,
        n_mean_activations_trials=100, n_indirect_effect_trials=25,
        prefixes=PREFIXES, separators=SEPARATORS, compute_baseline=compute_baseline,
        generate_str=False, metric='f1_score', universal_set=False, revision=None,
    )


def best_layer_from_zs(zs_path):
    with open(zs_path) as f:
        zs = json.load(f)
    best_L, best_acc = None, -1.0
    for L, entry in zs.items():
        topk = entry.get('intervention_topk') if isinstance(entry, dict) else None
        acc = dict(topk).get(1, 0.0) if topk else 0.0
        if acc > best_acc:
            best_acc, best_L = acc, int(L)
    return best_L, best_acc


def run_main_tasks(model, tokenizer, model_config, model_name, model_root):
    tag = tag_of(model_name)
    for task in TASKS:
        done_marker = os.path.join(model_root, task, 'model_baseline.json')
        if os.path.exists(done_marker):
            log(f"SKIP {tag} {task} main (cached)")
            continue
        ss = skip_shuffled_now()
        t = time.time()
        log(f"START {tag} {task} main skip_shuffled={ss}")
        try:
            args = make_args(task, model_name, model_root, model_root, edit_layer=-1,
                             n_top_heads=DEFAULT_TOP_HEADS, compute_baseline=True)
            run_evaluation(model, tokenizer, model_config, args, skip_shuffled=ss)
            bL, bAcc = best_layer_from_zs(os.path.join(model_root, task, 'zs_results_layer_sweep.json'))
            log(f"DONE {tag} {task} main elapsed={time.time()-t:.0f}s bestL={bL} bestFVacc={bAcc:.4f}")
        except Exception as e:
            log(f"ERROR {tag} {task} main elapsed={time.time()-t:.0f}s :: {repr(e)}")


def run_kablation(model, tokenizer, model_config, model_name, model_root):
    tag = tag_of(model_name)
    for task in KABLATION_TASKS:
        zs_path = os.path.join(model_root, task, 'zs_results_layer_sweep.json')
        if not os.path.exists(zs_path):
            log(f"SKIP_KABL {tag} {task} (no main zs results)")
            continue
        bL, _ = best_layer_from_zs(zs_path)
        for K in KABLATION_KS:
            k_root = os.path.join(model_root, f'k{K}')
            marker = os.path.join(k_root, task, f'zs_results_editlayer_{bL}.json')
            if os.path.exists(marker):
                log(f"SKIP {tag} {task} k{K} (cached)")
                continue
            # Reuse the main run's filter set so we don't re-run the 10-shot filter eval.
            os.makedirs(os.path.join(k_root, task), exist_ok=True)
            src_fs = os.path.join(model_root, task, 'fs_results_layer_sweep.json')
            dst_fs = os.path.join(k_root, task, 'fs_results_layer_sweep.json')
            if os.path.exists(src_fs) and not os.path.exists(dst_fs):
                shutil.copy(src_fs, dst_fs)
            ss = skip_shuffled_now()
            t = time.time()
            log(f"START {tag} {task} k{K} editlayer={bL} skip_shuffled={ss}")
            try:
                args = make_args(task, model_name, k_root, model_root, edit_layer=bL,
                                 n_top_heads=K, compute_baseline=False)
                run_evaluation(model, tokenizer, model_config, args, skip_shuffled=ss)
                log(f"DONE {tag} {task} k{K} elapsed={time.time()-t:.0f}s")
            except Exception as e:
                log(f"ERROR {tag} {task} k{K} elapsed={time.time()-t:.0f}s :: {repr(e)}")


def main():
    torch.set_grad_enabled(False)
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    if not os.path.exists(CONTROL_FILE):
        with open(CONTROL_FILE, 'w') as f:
            json.dump({'skip_shuffled': False}, f)
    log(f"SWEEP_START models={MODELS}")

    for model_name in MODELS:
        tag = tag_of(model_name)
        model_root = os.path.join(RESULTS_ROOT, tag)
        t0 = time.time()
        log(f"LOAD_START {model_name}")
        try:
            model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device='cuda')
        except Exception as e:
            log(f"LOAD_ERROR {model_name} :: {repr(e)}")
            continue
        log(f"LOAD_DONE {model_name} elapsed={time.time()-t0:.0f}s "
            f"n_layers={model_config['n_layers']} prepend_bos={model_config['prepend_bos']}")

        run_main_tasks(model, tokenizer, model_config, model_name, model_root)
        run_kablation(model, tokenizer, model_config, model_name, model_root)

        log(f"MODEL_DONE {tag}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    log("SWEEP_DONE")


if __name__ == '__main__':
    main()
