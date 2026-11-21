"""Figures and tables for the report, built from the processed data and the saved results.

Run from code/: .venv/bin/python make_report_figures.py
Writes PDFs to materials/report/img/ and LaTeX tables to materials/report/includes/tab_*.tex.
"""

import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(CODE, "..", "materials", "report")
sys.path.insert(0, CODE)
os.chdir(CODE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from modules import data_helper as dh, explain_helper as eh, features as ft, models as md
from modules import results_helper as rh

OUT = os.path.join(REPORT, "img")
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
    "logreg": "TF-IDF + LR", "bert_distilbert-base-uncased": "DistilBERT (fine-tuned)",
    "ocsvm": "One-class SVM", "dummy_majority": "Majority dummy",
    "stack_bert_logreg": "Stack: DistilBERT + TF-IDF LR",
}
SHORT = {"stack_bert_logreg": "Stack", "logreg": "TF-IDF LR", "bert_distilbert-base-uncased": "DistilBERT"}
MAIN = list(NAMES)
FAMILY = {"logreg": "classical", "bert_distilbert-base-uncased": "deep", "ocsvm": "no labels",
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
    with open(os.path.join(REPORT, "includes", "tab_headline.tex"), "w") as f:
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
    with open(os.path.join(REPORT, "includes", "tab_paired.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_headline.tex, tab_paired.tex")


def _tex(w):
    """Escape the characters LaTeX treats specially in a table cell."""
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")):
        w = w.replace(a, b)
    return w


def write_term_table(k=10):
    """Top coefficients of TF-IDF LR, refitted on the seed-0 training rows with its tuned settings."""
    df = ft.add_engineered_features(dh.load_processed())
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
    with open(os.path.join(REPORT, "includes", "tab_terms.tex"), "w") as f:
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
        # One entry per word: a word that occurs several times gets the sum of its attributions.
        total = {}
        for a, w in zip(attrs, words):
            # Keep words (and "$"); drop punctuation, stop words and one-letter fragments.
            if (w == "$" or (len(w) > 1 and any(c.isalnum() for c in w))) and w not in ENGLISH_STOP_WORDS:
                total[w] = total.get(w, 0.0) + a
        ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:k]
        top = [w if len(w) <= 16 else w[:8] + "..." for w, a in ranked if a > 0]
        title = " ".join(str(titles[r["job_id"]]).split())
        title = title if len(title) <= 26 else title[:24] + "..."
        lines.append(f"{r['kind']} & {_tex(title)} & {r['score']:.3f} & {sum(attrs):+.2f} & "
                     f"{_tex(', '.join(top))} \\\\")
    with open(os.path.join(REPORT, "includes", "tab_ig.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tab_ig.tex")


def fig_eda():
    """Four exploration panels on the raw data (as in notebook 01): balance, duplicates, length, country."""
    df = ft.add_engineered_features(dh.load_raw("../data/fake_job_postings.csv"))
    y = df["fraudulent"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    # (a) Class balance.
    ax = axes[0, 0]
    counts = y.value_counts().reindex([0, 1])
    ax.barh(["real", "fraud"], counts.values, color=[BLUE, ORANGE], height=0.6)
    for i, n in enumerate(counts.values):
        ax.text(n + 200, i, f"{n:,} ({n / len(y):.1%})", va="center", color=INK, fontsize=8)
    ax.set_xlim(0, 23000)
    ax.set_title("(a) Class balance", loc="left", fontsize=9)
    ax.set_xlabel("ads")
    ax.grid(axis="y", visible=False)

    # (b) Ads by the size of their description group (normalised description, as notebook 01).
    ax = axes[0, 1]
    desc = (df["description"].fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())
    size = desc.map(desc.value_counts())
    bins = [("1", size == 1), ("2-5", size.between(2, 5)), ("6-50", size.between(6, 50)),
            ("51+", size > 50)]
    labels, n_ads = [b[0] for b in bins], [int(m.sum()) for _, m in bins]
    ax.bar(labels, n_ads, color=BLUE, width=0.6)
    for i, n in enumerate(n_ads):
        ax.text(i, n * 1.15, f"{n:,}", ha="center", color=INK, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(100, 60000)
    ax.set_title("(b) Ads by size of their description group", loc="left", fontsize=9)
    ax.set_xlabel("ads sharing the same description")
    ax.set_ylabel("ads (log scale)")
    ax.grid(axis="x", visible=False)

    # (c) Description length by class (densities, log x).
    ax = axes[1, 0]
    lens = df["description"].fillna("").str.len()
    edges = np.logspace(1, np.log10(lens.max()), 35)
    for label, c, name in ((0, BLUE, "real"), (1, ORANGE, "fraud")):
        v = lens[(y == label) & (lens > 0)]
        # Each bin as a share of the class's ads, so both classes are comparable on a log axis.
        ax.hist(v, bins=edges, weights=np.full(len(v), 1 / len(v)), histtype="step",
                color=c, linewidth=1.8, label=name)
    ax.set_xscale("log")
    ax.set_title("(c) Description length", loc="left", fontsize=9)
    ax.set_xlabel("characters (log scale)")
    ax.set_ylabel("share of the class's ads")
    ax.legend(loc="upper left")

    # (d) Fraud rate by country (countries with at least 100 ads).
    ax = axes[1, 1]
    rate = df.groupby("country")["fraudulent"].agg(["size", "mean"])
    rate = rate[rate["size"] >= 100].sort_values("mean")
    ax.barh(rate.index, rate["mean"], color=ORANGE, height=0.6)
    ax.axvline(y.mean(), color=GRAY, linestyle="--", linewidth=1)
    ax.text(y.mean() + 0.004, -0.9, "overall", color=MUTED, fontsize=7)
    ax.set_title("(d) Fraud rate by country (100+ ads)", loc="left", fontsize=9)
    ax.set_xlabel("share of fraudulent ads")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "eda_overview.pdf")


if __name__ == "__main__":
    res = rh.load_results("outputs/classical", "outputs/deep", "outputs/stack")
    # The report's tables and the per-seed figure show only the models the report is about;
    # the other models are reported in the text.
    res = res[res["model"].isin(MAIN)]
    fig_eda()
    fig_flags()
    fig_seeds(res)
    write_tables(res)
    write_ig_table()
    write_term_table()
