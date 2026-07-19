#!/usr/bin/env python
"""
Stage 3 Part B: paper-ready tables (T1-T4) and figures (F1, F1b, F2).

Every number is read from committed files under results_repro/ (provenance: each
table's source files are listed in analysis/provenance.md).

Outputs:
  analysis/tables/T{1..4}_*.md      human-readable markdown
  analysis/tables/T{1..4}_*.tex     LaTeX (booktabs)
  analysis/figures/F1_layer_sweep.pdf        cross-model layer sweep, fractional depth (centerpiece)
  analysis/figures/F1b_layer_sweep_facets.pdf  per-task facets
  analysis/figures/F2_aie_concentration.pdf  cumulative AIE share vs head rank
  analysis/results_summary.json     all computed numbers in one place

Usage: python analysis/make_tables.py   (from repo root)
"""
import json, os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
R = os.path.join(REPO, 'results_repro')
TDIR = os.path.join(REPO, 'analysis', 'tables')
FDIR = os.path.join(REPO, 'analysis', 'figures')
os.makedirs(TDIR, exist_ok=True); os.makedirs(FDIR, exist_ok=True)

MODELS = [  # (results tag, display name, n_layers)
    ('EleutherAI_gpt-j-6b', 'GPT-J-6B', 28),
    ('meta-llama_Llama-3.1-8B', 'Llama-3.1-8B', 32),
    ('Qwen_Qwen2.5-7B', 'Qwen2.5-7B', 28),
    ('mistralai_Mistral-7B-v0.3', 'Mistral-7B-v0.3', 32),
]
TASKS = ['antonym', 'capitalize', 'country-capital', 'english-french', 'present-past', 'singular-plural']

# Todd et al. (ICLR 2024), Table 6 (arXiv 2310.15213v2): GPT-J zero-shot accuracy with
# the task FV (v_t column), mean +/- std over seeds, at the paper's chosen layer.
PAPER_GPTJ_FV = {'antonym': (48.2, 2.0), 'capitalize': (70.5, 2.4), 'country-capital': (83.2, 2.7),
                 'english-french': (44.7, 1.2), 'present-past': (19.7, 5.9), 'singular-plural': (47.0, 3.4)}

# Fixed model->style assignment used in EVERY figure (validated categorical order;
# marker + linestyle are secondary encodings so identity is never color-alone).
STYLE = {
    'GPT-J-6B':        dict(color='#2a78d6', marker='o', ls='-'),
    'Llama-3.1-8B':    dict(color='#008300', marker='s', ls='--'),
    'Qwen2.5-7B':      dict(color='#e87ba4', marker='^', ls='-.'),
    'Mistral-7B-v0.3': dict(color='#eda100', marker='D', ls=':'),
}
INK, MUTED, GRID = '#333333', '#666666', '#dddddd'


def top1(entry, which):
    d = dict(entry.get(which) or [])
    return d.get(1, 0.0)


def load_sweep(tag, task):
    zs = json.load(open(f'{R}/{tag}/{task}/zs_results_layer_sweep.json'))
    layers = sorted(int(L) for L in zs)
    clean = np.array([top1(zs[str(L)], 'clean_topk') for L in layers])
    interv = np.array([top1(zs[str(L)], 'intervention_topk') for L in layers])
    return layers, clean, interv


def write_table(name, headers, rows, caption):
    md = f"**{caption}**\n\n|" + "|".join(headers) + "|\n|" + "|".join(['---'] * len(headers)) + "|\n"
    for r in rows:
        md += "|" + "|".join(str(x) for x in r) + "|\n"
    open(f'{TDIR}/{name}.md', 'w').write(md)
    ncols = len(headers)
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n"
           f"\\begin{{tabular}}{{l{'r' * (ncols - 1)}}}\n\\toprule\n"
           + " & ".join(h.replace('%', '\\%').replace('#', '\\#') for h in headers) + " \\\\\n\\midrule\n")
    for r in rows:
        tex += " & ".join(str(x).replace('%', '\\%').replace('#', '\\#') for x in r) + " \\\\\n"
    tex += ("\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\end{{table}}\n")
    open(f'{TDIR}/{name}.tex', 'w').write(tex)
    print(f"wrote {name} (.md/.tex)")


summary = {}

# ---------------- T1: GPT-J reproduction vs paper ----------------
rows = []
t1_data = {}
for t in TASKS:
    layers, clean, interv = load_sweep('EleutherAI_gpt-j-6b', t)
    eff = interv - clean
    bi = int(np.argmax(eff))
    p_mean, p_std = PAPER_GPTJ_FV[t]
    rows.append([t, f"{p_mean:.1f}$\\pm${p_std:.1f}" if False else f"{p_mean:.1f}±{p_std:.1f}",
                 f"{interv[bi]*100:.1f}", f"L{layers[bi]}", f"{clean[bi]*100:.1f}", f"+{eff[bi]*100:.1f}"])
    t1_data[t] = {'paper_fv': p_mean, 'ours_fv': interv[bi] * 100, 'best_layer': layers[bi],
                  'clean': clean[bi] * 100, 'effect': eff[bi] * 100}
write_table('T1_gptj_reproduction',
            ['Task', 'Paper +FV (%)', 'Ours +FV (%)', 'Best layer', 'No-interv (%)', 'FV effect (pp)'],
            rows,
            'T1: GPT-J-6B reproduction. Paper values are Todd et al. Table 6 (zero-shot accuracy '
            'with the task FV at their chosen layer, mean±std over seeds); ours are best-over-layer-sweep '
            '(single seed 42), so ours upper-bound the paper protocol.')
summary['T1'] = t1_data

# ---------------- T2: generalization (FV effect matrix) ----------------
rows, t2_data = [], {}
for t in TASKS:
    row = [t]
    t2_data[t] = {}
    for tag, name, nl in MODELS:
        layers, clean, interv = load_sweep(tag, t)
        eff = interv - clean
        bi = int(np.argmax(eff))
        row.append(f"+{eff[bi]:.2f} (L{layers[bi]})")
        t2_data[t][name] = {'effect': round(float(eff[bi]), 3), 'best_layer': layers[bi],
                            'fv_acc': round(float(interv[bi]), 3), 'clean': round(float(clean[bi]), 3)}
    rows.append(row)
n_transfer = ['tasks transferring (effect$>$0.3)']
for tag, name, nl in MODELS:
    n = sum(1 for t in TASKS if t2_data[t][name]['effect'] > 0.3)
    n_transfer.append(f"{n}/6")
rows.append(n_transfer)
write_table('T2_generalization', ['Task'] + [m[1] for m in MODELS], rows,
            'T2: Zero-shot FV effect (accuracy with FV minus no-intervention accuracy, top-1) at the '
            'best-effect layer, per model and task.')
summary['T2'] = t2_data

# ---------------- T3: k-ablation ----------------
rows, t3_data = [], {}
KS = [5, 10, 20, 50, 100]
for tag, name, nl in MODELS:
    for t in ['antonym', 'country-capital']:
        zs = json.load(open(f'{R}/{tag}/{t}/zs_results_layer_sweep.json'))
        bL = max(zs, key=lambda L: top1(zs[L], 'intervention_topk'))
        vals = {}
        for K in KS:
            if K == 10:
                vals[K] = top1(zs[bL], 'intervention_topk')
            else:
                f = f'{R}/{tag}/k{K}/{t}/zs_results_editlayer_{bL}.json'
                if os.path.exists(f):
                    vals[K] = top1(json.load(open(f)), 'intervention_topk')
        row = [name, t, f"L{bL}"] + [f"{vals[K]:.2f}" if K in vals else '--' for K in KS]
        rows.append(row)
        t3_data[f'{name}/{t}'] = {'layer': int(bL), **{f'k{K}': round(float(v), 3) for K, v in vals.items()}}
write_table('T3_k_ablation', ['Model', 'Task', 'Layer', 'k=5', 'k=10', 'k=20', 'k=50', 'k=100'], rows,
            'T3: Zero-shot accuracy with FV built from top-k heads, injected at the best layer from '
            'the main (k=10) sweep. k=100 run for Mistral antonym only; dashes = not run.')
summary['T3'] = t3_data

# ---------------- T4: AIE concentration ----------------
rows, t4_data = [], {}
aie_sorted = {}
for tag, name, nl in MODELS:
    shares, shares1pct, maxes = [], [], []
    per_task_sorted = []
    for t in TASKS:
        ie = torch.load(f'{R}/{tag}/{t}/{t}_indirect_effect.pt', weights_only=True)
        m = ie.mean(0).flatten()
        pos_total = m.clamp(min=0).sum()
        srt = torch.sort(m, descending=True).values
        shares.append((srt[:10].sum() / pos_total).item())
        k1 = max(1, round(0.01 * m.numel()))
        shares1pct.append((srt[:k1].sum() / pos_total).item())
        maxes.append(m.max().item())
        per_task_sorted.append((srt.clamp(min=0) / pos_total).numpy())
    n_heads = m.numel()
    rows.append([name, n_heads, f"{np.mean(shares):.2f}", f"{np.mean(shares1pct):.2f}", f"{np.mean(maxes):.3f}"])
    t4_data[name] = {'n_heads': int(n_heads), 'top10_share_mean': round(float(np.mean(shares)), 3),
                     'top1pct_share_mean': round(float(np.mean(shares1pct)), 3),
                     'max_aie_mean': round(float(np.mean(maxes)), 4),
                     'top10_share_per_task': {t: round(s, 3) for t, s in zip(TASKS, shares)}}
    # average task-normalized sorted curve (pad to common length not needed; keep per model)
    L = min(len(c) for c in per_task_sorted)
    aie_sorted[name] = np.mean([c[:L] for c in per_task_sorted], axis=0)
write_table('T4_aie_concentration',
            ['Model', '# heads', 'Top-10 AIE share', 'Top-1% AIE share', 'Max per-head AIE'],
            rows,
            'T4: Concentration of causal task signal. Share of total positive mean AIE mass captured '
            'by the top-10 (and top-1\\% of) heads, and the largest single-head mean AIE, averaged over '
            'the six tasks.')
summary['T4'] = t4_data

# ---------------- F1: cross-model layer sweep (fractional depth) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)  # single column
for tag, name, nl in MODELS:
    effs = []
    for t in TASKS:
        layers, clean, interv = load_sweep(tag, t)
        effs.append(interv - clean)
    mean_eff = np.mean(effs, axis=0)
    x = np.array(layers) / nl
    st = STYLE[name]
    ax.plot(x, mean_eff, color=st['color'], ls=st['ls'], lw=1.6, marker=st['marker'],
            ms=3, markevery=4, label=name)
ax.set_xlabel('Injection layer (fraction of depth)', fontsize=8, color=INK)
ax.set_ylabel('Mean zero-shot FV effect', fontsize=8, color=INK)
ax.set_ylim(top=ax.get_ylim()[1] + 0.18)  # headroom so the legend clears Qwen's peak
ax.axvline(1 / 3, color=GRID, lw=1, zorder=0)
ax.text(1 / 3, ax.get_ylim()[1] * 0.97, ' L/3', fontsize=7, color=MUTED, va='top')
ax.tick_params(labelsize=7, colors=INK)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for s in ['left', 'bottom']:
    ax.spines[s].set_color(GRID)
ax.grid(axis='y', color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(f'{FDIR}/F1_layer_sweep.pdf')
plt.close(fig)
print('wrote F1_layer_sweep.pdf')

# ---------------- F1b: per-task facets ----------------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), dpi=200, sharex=True, sharey=True)
for ai, t in enumerate(TASKS):
    ax = axes.flat[ai]
    for tag, name, nl in MODELS:
        layers, clean, interv = load_sweep(tag, t)
        st = STYLE[name]
        ax.plot(np.array(layers) / nl, interv - clean, color=st['color'], ls=st['ls'],
                lw=1.3, marker=st['marker'], ms=2.5, markevery=5, label=name)
    ax.set_title(t, fontsize=8, color=INK)
    ax.axvline(1 / 3, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(labelsize=7, colors=INK)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
axes.flat[0].legend(fontsize=6, frameon=False)
fig.supxlabel('Injection layer (fraction of depth)', fontsize=9, color=INK)
fig.supylabel('Zero-shot FV effect', fontsize=9, color=INK)
fig.tight_layout()
fig.savefig(f'{FDIR}/F1b_layer_sweep_facets.pdf')
plt.close(fig)
print('wrote F1b_layer_sweep_facets.pdf')

# ---------------- F2: cumulative AIE share vs head rank ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=200)
for tag, name, nl in MODELS:
    curve = np.cumsum(aie_sorted[name])[:100]
    st = STYLE[name]
    ax.plot(np.arange(1, len(curve) + 1), curve, color=st['color'], ls=st['ls'], lw=1.6,
            marker=st['marker'], ms=3, markevery=[9], label=name)  # marker at rank 10
ax.axvline(10, color=GRID, lw=1, zorder=0)
ax.text(10, 0.02, ' top-10', fontsize=7, color=MUTED)
ax.set_xscale('log')
ax.set_xlabel('Head rank (log)', fontsize=8, color=INK)
ax.set_ylabel('Cumulative share of positive AIE', fontsize=8, color=INK)
ax.set_ylim(0, 1.0)
ax.tick_params(labelsize=7, colors=INK)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.grid(axis='y', color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(f'{FDIR}/F2_aie_concentration.pdf')
plt.close(fig)
print('wrote F2_aie_concentration.pdf')

json.dump(summary, open(os.path.join(REPO, 'analysis', 'results_summary.json'), 'w'), indent=2)
print('wrote results_summary.json')
