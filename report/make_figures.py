"""Figures for the report, built only from the saved results (no model is fitted).

Run from the repo root: code/.venv/bin/python report/make_figures.py
Writes PDFs to report/img/ and LaTeX tables to report/includes/tab_*.tex.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
os.chdir(CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from modules import data_helper as dh, explain_helper as eh, features as ft, models as md
from modules import results_helper as rh

OUT = os.path.join(HERE, "img")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})

# Display names and families, one name per model for the whole report.
NAMES = {
    "logreg_bm25": "BM25 + LR", "mlp": "TF-IDF + MLP", "logreg": "TF-IDF + LR",
    "bert_distilbert-base-uncased": "DistilBERT (fine-tuned)", "cnb": "Complement NB",
    "emb_logreg": "MiniLM embeddings + LR",
    "zeroshot_Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B (zero-shot)", "ocsvm": "One-class SVM",
    "dummy_majority": "Majority dummy", "stack_bert_logreg": "Stack: DistilBERT + TF-IDF LR",
}
SHORT = {"stack_bert_logreg": "Stack", "logreg_bm25": "BM25 LR", "logreg": "TF-IDF LR",
         "bert_distilbert-base-uncased": "DistilBERT", "mlp": "MLP", "cnb": "Compl. NB"}
MAIN = ["stack_bert_logreg", "logreg", "bert_distilbert-base-uncased", "ocsvm", "dummy_majority"]
FAMILY = {"logreg_bm25": "classical", "mlp": "classical", "logreg": "classical", "cnb": "classical",
          "bert_distilbert-base-uncased": "deep", "emb_logreg": "deep",
          "zeroshot_Qwen2.5-1.5B-Instruct": "no labels", "ocsvm": "no labels",
          "dummy_majority": "no labels", "stack_bert_logreg": "stack"}
FAMILY_COLOR = {"classical": BLUE, "deep": ORANGE, "stack": AQUA, "no labels": GRAY}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_flags():
    """Share of ads where each engineered flag fires, real vs fraud (raw data, as in notebook 01)."""
    df = ft.add_engineered_features(dh.load_raw("../data/fake_job_postings.csv"))
    flags = ["missing_company_profile", "has_company_logo", "has_money_words", "has_questions",
             "has_email", "has_salary", "has_phone", "telecommuting"]
    share = df.groupby("fraudulent")[flags].mean().T
    share = share.reindex((share[1] - share[0]).abs().sort_values().index)
    y = np.arange(len(share))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.barh(y + 0.2, share[0], 0.38, color=BLUE, label="real", edgecolor="white", linewidth=1)
    ax.barh(y - 0.2, share[1], 0.38, color=ORANGE, label="fraud", edgecolor="white", linewidth=1)
    ax.set_yticks(y, [f.replace("_", " ") for f in share.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of ads where the flag is set")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "eda_flags.pdf")


def fig_seeds(res):
    """Test PR-AUC of every model on every seed; the bar marks the mean."""
    order = res.groupby("model")["test_pr_auc"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(order) + 0.9))
    for i, m in enumerate(order):
        v = res.loc[res["model"] == m, "test_pr_auc"].to_numpy()
        c = FAMILY_COLOR[FAMILY[m]]
        ax.scatter(v, np.full(len(v), i), s=22, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.5, zorder=4)
        ax.text(1.02, i, f"{v.mean():.3f}", va="center", color=INK, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.axvline(res.loc[res["model"] == "dummy_majority", "test_pr_auc"].mean(), color=GRAY,
               linestyle="--", linewidth=1)
    ax.set_yticks(range(len(order)), [NAMES[m] for m in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("test PR-AUC (dots: 5 split seeds, bar: mean; dashed: fraud rate)")
    ax.grid(axis="y", visible=False)
    for fam, c in FAMILY_COLOR.items():
        ax.scatter([], [], s=22, color=c, label=fam)
    ax.legend(loc="center", bbox_to_anchor=(0.38, 0.5), title=None)  # the empty middle of the axis
    save(fig, "test_pr_auc_seeds.pdf")


def write_tables(res):
    """Headline table (mean +- std over seeds) and paired per-seed differences, as LaTeX rows."""
    order = res.groupby("model")["test_pr_auc"].mean().sort_values(ascending=False).index
    lines = []
    for m in order:
        g = res[res["model"] == m]
        cells = [f"{g[c].mean():.3f} $\\pm$ {g[c].std():.3f}" for c in
                 ("test_pr_auc", "test_roc_auc", "test_f1", "test_precision", "test_recall")]
        sec = g["seconds"].mean()
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + (f" & {sec:.0f} \\\\" if sec == sec else " & -- \\\\"))
    with open(os.path.join(HERE, "includes", "tab_headline.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    t = res.pivot(index="seed", columns="model", values="test_pr_auc")
    pairs = [("stack_bert_logreg", "logreg"), ("stack_bert_logreg", "bert_distilbert-base-uncased"),
             ("bert_distilbert-base-uncased", "logreg")]
    lines = []
    for a, b in pairs:
        d = t[a] - t[b]
        wins = int((d > 0).sum())
        lines.append(f"{SHORT[a]} $-$ {SHORT[b]} & " + " & ".join(f"{x:+.3f}" for x in d)
                     + f" & {d.mean():+.3f} & {wins}/5 \\\\")
    with open(os.path.join(HERE, "includes", "tab_paired.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_headline.tex, tab_paired.tex")


def _tex(w):
    """Escape the characters LaTeX treats specially in a table cell."""
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")):
        w = w.replace(a, b)
    return w


def write_term_table(k=10):
    """Top coefficients of TF-IDF LR, refitted on the seed-0 training rows with its tuned settings."""
    df = ft.add_engineered_features(dh.clean(dh.load_raw("../data/fake_job_postings.csv")))
    train = df.iloc[dh.load_splits("splits/seed0.npz", df)["train"]]
    with open("outputs/classical/logreg_seed0.json") as f:
        params = {k_: tuple(v) if isinstance(v, list) else v for k_, v in json.load(f)["best_params"].items()}
    pipe = md.get_models(0)["logreg"].set_params(**params).fit(train, train["fraudulent"])
    t = eh.top_linear_terms(pipe, k=k)

    def name(term):
        block, _, value = term.partition("__")
        return value.replace("_", ": ", 1) if block == "nominal" else value

    pos = t[t["direction"] == "positive"].reset_index(drop=True)
    neg = t[t["direction"] == "negative"].reset_index(drop=True)
    lines = [f"{_tex(name(p.term))} & {p.coef:+.2f} & {_tex(name(n.term))} & {n.coef:+.2f} \\\\"
             for p, n in zip(pos.itertuples(), neg.itertuples())]
    with open(os.path.join(HERE, "includes", "tab_terms.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_terms.tex")


def write_ig_table(k=5):
    """Top Integrated Gradients words per explained ad (stop words and punctuation dropped)."""
    with open("outputs/deep/explain/bert_distilbert-base-uncased_seed0.json") as f:
        rows = json.load(f)
    titles = dh.load_raw("../data/fake_job_postings.csv").set_index("job_id")["title"]
    lines = []
    for r in rows:
        words, attrs = [], []
        for t, a in zip(r["tokens"], r["attributions"]):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if t.startswith("##") and words:
                words[-1] += t[2:]
                attrs[-1] += a
            else:
                words.append(t)
                attrs.append(a)
        keep = [(a, w) for a, w in zip(attrs, words)
                if (w == "$" or any(c.isalnum() for c in w)) and w not in ENGLISH_STOP_WORDS]
        top = [w if len(w) <= 16 else w[:8] + "..." for a, w in sorted(keep, reverse=True)[:k] if a > 0]
        title = str(titles[r["job_id"]]).strip()
        title = title if len(title) <= 26 else title[:24] + "..."
        lines.append(f"{r['kind']} & {_tex(title)} & {r['score']:.3f} & {sum(attrs):+.2f} & "
                     f"{_tex(', '.join(top))} \\\\")
    with open(os.path.join(HERE, "includes", "tab_ig.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_ig.tex")


if __name__ == "__main__":
    res = rh.load_results("outputs/classical", "outputs/deep", "outputs/stack")
    # The report's tables and the per-seed figure show only the models the report is about;
    # the other models are reported in the text.
    res = res[res["model"].isin(MAIN)]
    fig_flags()
    fig_seeds(res)
    write_tables(res)
    write_ig_table()
    write_term_table()
