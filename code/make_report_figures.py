"""Generate current report figures and LaTeX tables from validated five-seed artifacts.

Run from the repository root: .venv/bin/python code/make_report_figures.py
The script never trains a model except seed-0 LR/tree refits used for explanations.
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
import pandas as pd
from modules import data_helper as dh
from modules import eval_helper as ev
from modules import explain_helper as eh
from modules import features as ft
from modules import models as md
from modules import results_helper as rh
from sklearn.metrics import precision_recall_curve

OUT = os.path.join(REPORT, "img")
INCLUDES = os.path.join(REPORT, "includes")
LATEX_END = r" \\"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
FAMILY_COLOR = {
    "classical": BLUE,
    "finetuned_top2": AQUA,
    "finetuned": "#8c5bc4",
}
FAMILY_TITLE = {
    "classical": "Classical TF-IDF",
    "finetuned_top2": "Top-two fine-tuning",
    "finetuned": "Full fine-tuning",
}
RUN_DIRS = {
    "classical": "outputs/classical",
    "finetuned_top2": "outputs/distilbert_top2",
    "finetuned": "outputs/finetuned",
}
FAMILY_MODELS = {
    "classical": ("logreg", "logreg_text", "tree", "ocsvm", "dummy_majority"),
    "finetuned_top2": ("distilbert_top2_text", "distilbert_top2_fused"),
    "finetuned": ("distilbert_text", "distilbert_fused"),
}
NAMES = {
    "dummy_majority": "Majority dummy",
    "logreg_text": "Text-only LR",
    "logreg": "Structured LR",
    "tree": "TF-IDF tree",
    "ocsvm": "TF-IDF one-class SVM",
    "distilbert_top2_text": "Top-two fine-tuned text",
    "distilbert_top2_fused": "Top-two fine-tuned fusion",
    "distilbert_text": "Full fine-tuned text",
    "distilbert_fused": "Full fine-tuned fusion",
}
EXPECTED = set(NAMES)
SEEDS = tuple(range(5))
MARKERS = ("o", "s", "^", "D", "v")
PRIMARY_PR_MODELS = (
    "logreg",
    "logreg_text",
    "distilbert_top2_text",
    "distilbert_top2_fused",
    "distilbert_text",
    "distilbert_fused",
)
ADAPTATION_ARMS = (
    ("Text-only", "distilbert_text", "distilbert_top2_text"),
    ("Structured fusion", "distilbert_fused", "distilbert_top2_fused"),
)
FUSION_ARMS = (
    ("Top-two fine-tuning", "distilbert_top2_fused", "distilbert_top2_text"),
    ("Full fine-tuning", "distilbert_fused", "distilbert_text"),
)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    """Save one vector figure and close its Matplotlib state."""
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def tex(value):
    """Escape text used inside generated LaTeX table cells."""
    value = str(value)
    for old, new in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                     ("#", "\\#"), ("_", "\\_")):
        value = value.replace(old, new)
    return value


def pm(series):
    """Format mean and sample standard deviation for one result column."""
    return f"${series.mean():.3f} \\pm {series.std():.3f}$"


def paired_delta(res, left, right, split, metric):
    """Return one ordered left-minus-right difference per matched split."""
    paired = res[res["model"].isin([left, right])].pivot(
        index="seed", columns="model", values=f"{split}_{metric}"
    )
    return (paired[left] - paired[right]).reindex(SEEDS)


def load_validated_results():
    """Load the exact 45-run matrix and recompute its metrics from raw scores."""
    res = rh.load_results(*RUN_DIRS.values())
    if len(res) != 45 or set(res["model"]) != EXPECTED:
        raise ValueError(f"incomplete or stale model matrix: {len(res)} rows, {set(res['model'])}")

    for family, expected in FAMILY_MODELS.items():
        rows = res[res["group"] == family]
        if len(rows) != 5 * len(expected) or set(rows["model"]) != set(expected):
            raise ValueError(f"incomplete or stale {family} runs")
    for model, rows in res.groupby("model"):
        if len(rows) != len(SEEDS) or set(rows["seed"]) != set(SEEDS):
            raise ValueError(f"incomplete or duplicate seeds for {model}")

    frame = dh.load_processed()
    fingerprint = dh.data_fingerprint(frame)
    split_cache = {seed: dh.load_splits(f"splits/seed{seed}.npz", frame) for seed in SEEDS}
    metrics = ("pr_auc", "roc_auc", "f1", "precision", "recall")
    for row in res.itertuples(index=False):
        with open(os.path.join(RUN_DIRS[row.group], f"{row.model}_seed{row.seed}.json")) as source:
            record = json.load(source)
        scores = rh.load_scores(RUN_DIRS[row.group], row.model, row.seed)
        rh.assert_result_lineage(
            record, scores, fingerprint, split_cache[row.seed]["val"], split_cache[row.seed]["test"]
        )
        for split in ("val", "test"):
            computed = ev.evaluate(
                scores[f"y_{split}"], scores[f"score_{split}"], row.thr
            )
            for metric in metrics:
                saved = getattr(row, f"{split}_{metric}")
                if not np.isclose(saved, computed[metric], rtol=1e-10, atol=1e-12):
                    raise ValueError(
                        f"stale {split}_{metric} for {row.model} seed {row.seed}: "
                        f"{saved} != {computed[metric]}"
                    )
    print("validated data/split lineage and recomputed headline metrics for 45 runs", flush=True)
    return res


def fig_eda_overview(df):
    """Class balance, related groups, and text-length distributions on cleaned data."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))

    counts = df["fraudulent"].value_counts().sort_index()
    ax = axes[0, 0]
    ax.bar(["real", "fraud"], counts.values, color=[BLUE, ORANGE])
    for index, value in enumerate(counts.values):
        ax.text(index, value + 200, f"{value:,}\n({value / len(df):.1%})", ha="center")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("(a) Class balance", loc="left")
    ax.set_ylabel("ads")
    ax.grid(axis="x", visible=False)

    sizes = df["group"].map(df["group"].value_counts())
    bins = [("1", sizes == 1), ("2–5", sizes.between(2, 5)),
            ("6–50", sizes.between(6, 50)), ("51+", sizes > 50)]
    values = [int(mask.sum()) for _, mask in bins]
    ax = axes[0, 1]
    ax.bar([label for label, _ in bins], values, color=BLUE)
    ax.set_yscale("log")
    ax.set_ylim(top=max(values) * 1.8)
    for index, value in enumerate(values):
        ax.text(index, value * 1.15, f"{value:,}", ha="center")
    ax.set_title("(b) Ads by related-group size", loc="left")
    ax.set_xlabel("ads per exact related-ad group")
    ax.set_ylabel("ads (log scale)")
    ax.grid(axis="x", visible=False)

    for ax, col, letter in ((axes[1, 0], "description", "c"),
                            (axes[1, 1], "requirements", "d")):
        lengths = df[col].fillna("").astype(str).str.len()
        positive = lengths[lengths > 0]
        edges = np.geomspace(max(1, positive.min()), positive.max(), 36)
        for label, color, name in ((0, BLUE, "real"), (1, ORANGE, "fraud")):
            subset = lengths[(df["fraudulent"] == label) & (lengths > 0)]
            ax.hist(subset, bins=edges, weights=np.full(len(subset), 100 / len(subset)),
                    histtype="step", linewidth=1.6, color=color, label=name)
        ax.set_xscale("log")
        ax.set_title(f"({letter}) {col.capitalize()} length", loc="left")
        ax.set_xlabel("characters (log scale)")
        ax.set_ylabel("% of nonempty ads in class per bin")
        ax.legend()
    fig.tight_layout()
    save(fig, "eda_overview.pdf")


def fig_eda_flags(dfe):
    """Presence of candidate binary features by class."""
    share = dfe.groupby("fraudulent")[ft.CANDIDATE_BINARY_COLS].mean().T
    share = share.loc[(share[1] - share[0]).abs().sort_values().index]
    y = np.arange(len(share))
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.barh(y - 0.2, share[0], 0.4, color=BLUE, label="real")
    ax.barh(y + 0.2, share[1], 0.4, color=ORANGE, label="fraud")
    ax.set_yticks(y, [name.replace("_", " ") for name in share.index])
    ax.set_xlabel("share of ads with the feature")
    ax.set_title("Candidate presence features on the cleaned data", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend()
    fig.tight_layout()
    save(fig, "eda_flags.pdf")


def fig_eda_categories(dfe):
    """Highest and lowest category fraud rates with at least 100 ads."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    overall = dfe["fraudulent"].mean()
    for ax, col in zip(axes.flat, ft.NOMINAL_COLS):
        rates = dfe.groupby(col, dropna=False)["fraudulent"].agg(n="size", fraud_rate="mean")
        rates = rates[rates["n"] >= 100]
        selected = pd.concat([rates.nlargest(2, "fraud_rate"), rates.nsmallest(2, "fraud_rate")])
        selected = selected.sort_values("fraud_rate")
        labels = ["(missing)" if pd.isna(value) else str(value)[:28] for value in selected.index]
        positions = np.arange(len(selected))
        ax.barh(positions, selected["fraud_rate"], color=ORANGE)
        ax.axvline(overall, color=GRAY, linestyle="--")
        ax.set_yticks(positions, labels)
        ax.set_title(col.replace("_", " "), loc="left")
        ax.set_xlabel("fraud rate")
        for position, (_, row) in zip(positions, selected.iterrows()):
            ax.text(row["fraud_rate"] + 0.004, position, f"n={int(row['n']):,}",
                    va="center", fontsize=7)
        ax.set_xlim(0, max(0.45, selected["fraud_rate"].max() + 0.08))
        ax.grid(axis="y", visible=False)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Highest and lowest category fraud rates (categories with 100+ ads)")
    fig.tight_layout()
    save(fig, "eda_categories.pdf")


def fig_seed_ap(res):
    """Per-seed test AP and model means in readable family facets."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    prevalence = res.loc[res["model"] == "dummy_majority", "test_pr_auc"].mean()
    for ax, (family, expected) in zip(axes.flat, FAMILY_MODELS.items()):
        means = res[res["group"] == family].groupby("model")["test_pr_auc"].mean()
        order = means.loc[list(expected)].sort_values().index
        color = FAMILY_COLOR[family]
        for position, model in enumerate(order):
            rows = res[res["model"] == model].sort_values("seed")
            x = position + np.linspace(-0.14, 0.14, len(rows))
            ax.scatter(x, rows["test_pr_auc"], color=color, s=25, zorder=3)
            ax.hlines(
                rows["test_pr_auc"].mean(), position - 0.24, position + 0.24,
                color=color, linewidth=2
            )
        ax.axhline(prevalence, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(order)), [NAMES[name] for name in order])
        ax.tick_params(axis="x", rotation=28, labelsize=8)
        ax.set_title(FAMILY_TITLE[family], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test AP")
    fig.suptitle("Test AP across five grouped splits")
    fig.text(
        0.5, 0.012,
        "Points: individual splits; horizontal ticks: model means; dashed lines: mean fraud prevalence.",
        ha="center", fontsize=8
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save(fig, "test_ap_by_seed.pdf")


def fig_validation_test(res):
    """Validation/test AP pairs for all 45 runs in family facets."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    for ax, (family, models) in zip(axes.flat, FAMILY_MODELS.items()):
        ax.plot([0, 0.9], [0, 0.9], color=GRAY, linestyle="--", linewidth=1)
        for model, marker in zip(models, MARKERS):
            rows = res[res["model"] == model]
            ax.scatter(
                rows["val_pr_auc"], rows["test_pr_auc"], s=30, alpha=0.78,
                color=FAMILY_COLOR[family], marker=marker, label=NAMES[model]
            )
        ax.set_xlim(0, 0.9)
        ax.set_ylim(0, 0.9)
        ax.set_title(FAMILY_TITLE[family], loc="left")
        ax.legend(loc="upper left", fontsize=7)
    fig.supxlabel("validation AP")
    fig.supylabel("test AP")
    fig.suptitle("Validation and test AP for 45 validated runs (dashed: equality)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "validation_vs_test_ap.pdf")


def fig_feature_ablation(res):
    """Structured-minus-text LR deltas in AP and F1."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, label in ((axes[0], "pr_auc", "AP"), (axes[1], "f1", "F1")):
        for split, color in (("val", AQUA), ("test", ORANGE)):
            paired = res[res["model"].isin(["logreg", "logreg_text"])].pivot(
                index="seed", columns="model", values=f"{split}_{metric}"
            )
            delta = paired["logreg"] - paired["logreg_text"]
            label_name = "validation" if split == "val" else "test"
            ax.plot(delta.index, delta.values, marker="o", color=color,
                    label=f"{label_name}; mean {delta.mean():+.3f}")
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_xticks(range(5))
        ax.set_xlabel("split seed")
        ax.set_ylabel(f"structured minus text-only {label}")
        ax.set_title(f"Change in {label}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "feature_ablation.pdf")


def fig_adaptation_ablation(res):
    """Paired full-minus-top-two AP deltas separately for validation and test."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        for (label, full, top2), color in zip(ADAPTATION_ARMS, (BLUE, ORANGE)):
            delta = paired_delta(res, full, top2, split, "pr_auc")
            ax.plot(
                delta.index, delta.values, marker="o", color=color,
                label=f"{label}; mean {delta.mean():+.3f}"
            )
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.set_xticks(SEEDS)
        ax.set_xlabel("split seed")
        ax.set_title("Validation" if split == "val" else "Test", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("full minus top-two $\\Delta$AP")
    fig.suptitle("Paired effect of full rather than top-two encoder fine-tuning")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "adaptation_ablation.pdf")


def fig_metric_profile(res):
    """Mean thresholded precision, recall, and F1 by non-dummy model and family."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True)
    means = res.groupby("model")[["test_pr_auc", "test_precision", "test_recall", "test_f1"]].mean()
    for ax, (family, expected) in zip(axes.flat, FAMILY_MODELS.items()):
        models = [name for name in expected if name != "dummy_majority"]
        order = means.loc[models, "test_pr_auc"].sort_values().index
        y = np.arange(len(order))
        for offset, metric, color, label in (
            (-0.24, "test_precision", BLUE, "precision"),
            (0.0, "test_recall", ORANGE, "recall"),
            (0.24, "test_f1", AQUA, "F1"),
        ):
            ax.barh(y + offset, means.loc[order, metric], height=0.22, color=color, label=label)
        ax.set_yticks(y, [NAMES[name] for name in order], fontsize=8)
        ax.set_title(FAMILY_TITLE[family], loc="left")
        ax.grid(axis="y", visible=False)
    for ax in axes:
        ax.set_xlabel("mean over five splits")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=3)
    fig.suptitle("Thresholded test metrics at validation-selected thresholds (8 non-dummy models)")
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    save(fig, "metric_profile.pdf")


def fig_pr_curves(res):
    """Seed-0 precision-recall curves for the stated primary model subset."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y_test = None
    seed0 = res[res["seed"] == 0].set_index("model")
    for name in PRIMARY_PR_MODELS:
        scores = rh.load_scores(RUN_DIRS[seed0.loc[name, "group"]], name, 0)
        if y_test is None:
            y_test = scores["y_test"]
        assert np.array_equal(y_test, scores["y_test"])
        precision, recall, _ = precision_recall_curve(y_test, scores["score_test"])
        ax.plot(recall, precision, label=NAMES[name], linewidth=1.4)
    ax.axhline(y_test.mean(), color=GRAY, linestyle="--", label="fraud rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Test precision-recall curves (seed 0): primary six-model subset", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.text(
        0.02, 0.012,
        "Subset: structured LR, text-only LR, and top-two/full text and fusion models.",
        fontsize=8
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "pr_curves_seed0.pdf")


def fig_confusion_matrices(res):
    """Verify and plot all seed-0 confusion matrices by technique family."""
    families = (
        (
            "Classical models", "confusion_classical.pdf", "Blues", (2, 3),
            (
                ("logreg_text", "Text-only LR"), ("logreg", "Structured LR"),
                ("tree", "TF-IDF tree"), ("ocsvm", "TF-IDF one-class SVM"),
                ("dummy_majority", "Majority dummy"),
            ),
        ),
        (
            "Deep-learning models", "confusion_deep.pdf", "Oranges", (2, 2),
            (
                ("distilbert_top2_text", "Top-two text"),
                ("distilbert_top2_fused", "Top-two fusion"),
                ("distilbert_text", "Full text"), ("distilbert_fused", "Full fusion"),
            ),
        ),
    )
    seed0 = res[res["seed"] == 0].set_index("model")
    states = (("TN", "FP"), ("FN", "TP"))
    for title, filename, color_map, shape, models in families:
        fig, axes = plt.subplots(*shape, figsize=(3.15 * shape[1], 5.8), squeeze=False)
        for ax, (name, label) in zip(axes.flat, models):
            scores = rh.load_scores(RUN_DIRS[seed0.loc[name, "group"]], name, 0)
            threshold = float(seed0.loc[name, "thr"])
            metrics = ev.evaluate(scores["y_test"], scores["score_test"], threshold)
            matrix = np.array([
                [metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]],
            ])
            with open(os.path.join(
                    RUN_DIRS[seed0.loc[name, "group"]], f"{name}_seed0.json")) as source:
                expected = json.load(source)["test"]
            assert np.array_equal(matrix, [
                [expected["tn"], expected["fp"]], [expected["fn"], expected["tp"]],
            ]), name
            assert matrix.sum() == len(scores["y_test"]) == 3520
            vmax = max(1, int(matrix[0].max()))
            ax.imshow(matrix, cmap=color_map, vmin=0, vmax=vmax)
            ax.set_xticks([0, 1], ["real", "fraud"])
            ax.set_yticks([0, 1], ["real", "fraud"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("actual")
            ax.set_title(label, fontsize=9)
            ax.grid(visible=False)
            for actual in (0, 1):
                for predicted in (0, 1):
                    count = int(matrix[actual, predicted])
                    ax.text(
                        predicted, actual, f"{states[actual][predicted]} {count:,}",
                        ha="center", va="center", fontsize=10,
                        color="white" if count > 0.62 * vmax else INK,
                    )
        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        fig.suptitle(f"Seed-0 test confusion matrices: {title}", fontsize=12)
        threshold_note = "Each model uses its own validation-F1 threshold."
        if any(model == "dummy_majority" for model, _ in models):
            threshold_note += " The dummy keeps its fixed 0.5 threshold."
        fig.text(0.5, 0.015, threshold_note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        save(fig, filename)


def write_result_tables(res):
    """Write family-headline and paired-ablation LaTeX tables."""
    families = (
        ("tab_classical_results.tex", ("classical",)),
        ("tab_deep_results.tex", ("finetuned_top2", "finetuned")),
    )
    for filename, groups in families:
        models = [model for group in groups for model in FAMILY_MODELS[group]]
        subset = res[res["model"].isin(models)]
        order = subset.groupby("model")["test_pr_auc"].mean().sort_values(ascending=False).index
        lines = []
        for name in order:
            rows = subset[subset["model"] == name]
            cells = [pm(rows[f"test_{metric}"]) for metric in ("pr_auc", "f1", "precision", "recall")]
            lines.append(f"{tex(NAMES[name])} & " + " & ".join(cells)
                         + f" & {rows['seconds'].mean():.1f}" + LATEX_END)
        with open(os.path.join(INCLUDES, filename), "w") as f:
            f.write("\n".join(lines) + "\n")

    lines = []
    for metric, label in (("pr_auc", "AP"), ("f1", "F1")):
        for split in ("val", "test"):
            paired = res[res["model"].isin(["logreg", "logreg_text"])].pivot(
                index="seed", columns="model", values=f"{split}_{metric}"
            )
            delta = paired["logreg"] - paired["logreg_text"]
            split_label = "validation" if split == "val" else "test"
            lines.append(f"{split_label} {label} & " + " & ".join(f"{x:+.3f}" for x in delta)
                         + f" & {delta.mean():+.3f} & {int((delta > 0).sum())}/5" + LATEX_END)
    with open(os.path.join(INCLUDES, "tab_paired.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    for filename, arms in (("tab_adaptation.tex", ADAPTATION_ARMS), ("tab_fusion.tex", FUSION_ARMS)):
        lines = []
        for label, left, right in arms:
            val_ap = paired_delta(res, left, right, "val", "pr_auc")
            test_ap = paired_delta(res, left, right, "test", "pr_auc")
            test_f1 = paired_delta(res, left, right, "test", "f1")
            lines.append(
                f"{tex(label)} & {val_ap.mean():+.3f} & {test_ap.mean():+.3f} & "
                f"{test_f1.mean():+.3f} & {int((test_ap > 0).sum())}/5 & "
                f"{int((test_f1 > 0).sum())}/5" + LATEX_END
            )
        with open(os.path.join(INCLUDES, filename), "w") as f:
            f.write("\n".join(lines) + "\n")
    print("wrote result tables", flush=True)


def write_explanation_tables(dfe):
    """Refit only the selected seed-0 classical models for coefficient/importance tables."""
    splits = dh.load_splits("splits/seed0.npz", dfe)
    train = dfe.iloc[splits["train"]]
    fitted = {}
    for name in ("logreg", "tree"):
        with open(f"outputs/classical/{name}_seed0.json", encoding="utf-8") as f:
            params = json.load(f)["best_params"]
        fitted[name] = md.get_models(0)[name].set_params(**params).fit(train, train["fraudulent"])

    terms = eh.top_linear_terms(fitted["logreg"], k=10)

    def label(term):
        block, _, value = term.partition("__")
        if block == "nominal":
            for field in ft.NOMINAL_COLS:
                prefix = field + "_"
                if value.startswith(prefix):
                    return f"{field.replace('_', ' ')}: {value[len(prefix):]}"
        return value.replace("_", " ")

    positive = terms[terms["direction"] == "positive"].reset_index(drop=True)
    negative = terms[terms["direction"] == "negative"].reset_index(drop=True)
    lines = [f"{tex(label(p.term))} & {p.coef:+.2f} & {tex(label(n.term))} & {n.coef:+.2f}" + LATEX_END
             for p, n in zip(positive.itertuples(), negative.itertuples())]
    with open(os.path.join(INCLUDES, "tab_terms.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    tree = fitted["tree"]
    importance = pd.Series(tree.named_steps["clf"].feature_importances_,
                           index=tree.named_steps["pre"].get_feature_names_out())
    top = importance.nlargest(10)
    lines = [f"{tex(label(name))} & {value:.3f}" + LATEX_END for name, value in top.items()]
    with open(os.path.join(INCLUDES, "tab_tree_terms.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote explanation tables", flush=True)


def main():
    """Generate every current report asset from one full artifact matrix."""
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(INCLUDES, exist_ok=True)
    res = load_validated_results()
    dfe = ft.add_engineered_features(dh.load_processed())
    df = dfe.drop(columns=["text"], errors="ignore")
    fig_eda_overview(df)
    fig_eda_flags(dfe)
    fig_eda_categories(dfe)
    fig_seed_ap(res)
    fig_validation_test(res)
    fig_feature_ablation(res)
    fig_adaptation_ablation(res)
    fig_metric_profile(res)
    fig_pr_curves(res)
    fig_confusion_matrices(res)
    write_result_tables(res)
    write_explanation_tables(dfe)


if __name__ == "__main__":
    main()
