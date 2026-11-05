"""Matplotlib figures in the SentinelPay look: midnight ink, periwinkle traffic, amber anomalies."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from .config import PALETTE as C
from .config import PCA_FEATURES


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["panel"], "savefig.facecolor": C["bg"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["muted"], "axes.titlecolor": C["text"],
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": C["grid"], "grid.alpha": 0.45, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": C["muted"], "ytick.color": C["muted"], "text.color": C["text"],
        "legend.facecolor": C["panel"], "legend.edgecolor": C["grid"], "legend.labelcolor": C["text"],
        "font.size": 10.5, "figure.dpi": 110, "savefig.dpi": 160,
    })


apply_theme()
GLOW = [pe.withStroke(linewidth=4, foreground=matplotlib.colors.to_rgba(C["anomaly"], 0.25))]


def save(fig, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# ------------------------------------------------------------------ EDA
def plot_class_imbalance(df: pd.DataFrame):
    counts = df["Class"].value_counts().sort_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    bars = a.bar(["Legitimate", "Fraud"], counts.values, color=[C["normal"], C["anomaly"]], width=0.55)
    a.set_yscale("log"); a.set_title("Transactions per class (log scale)"); a.grid(axis="x", visible=False)
    for bar, v in zip(bars, counts.values):
        a.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:,}", ha="center", fontweight="bold")
    a.set_ylim(top=counts.max() * 4)
    share = counts.values / counts.sum()
    wedges, _ = b.pie(share, colors=[C["normal"], C["anomaly"]], startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.32, edgecolor=C["bg"], linewidth=3))
    b.text(0, 0.06, f"{share[1] * 100:.2f}%", ha="center", fontsize=22, fontweight="bold", color=C["anomaly"])
    b.text(0, -0.2, "of traffic is fraud", ha="center", color=C["muted"], fontsize=10)
    b.set_title("Share of fraud"); b.grid(False)
    fig.tight_layout()
    return fig


def plot_amount_time(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    n, f = df[df.Class == 0], df[df.Class == 1]
    bins = np.logspace(np.log10(max(df.Amount.min(), 0.01)), np.log10(df.Amount.max()), 50)
    ax[0].hist(n.Amount, bins=bins, color=C["normal"], alpha=0.85, density=True, label="Legitimate")
    ax[0].hist(f.Amount, bins=bins, color=C["anomaly"], alpha=0.75, density=True, label="Fraud")
    ax[0].set_xscale("log"); ax[0].set_title("Amount distribution (log x)"); ax[0].set_xlabel("Amount"); ax[0].legend()
    hours = (df.Time % 86400) / 3600
    ax[1].hist(hours[df.Class == 0], bins=24, color=C["normal"], alpha=0.85, density=True)
    ax[1].hist(hours[df.Class == 1], bins=24, color=C["anomaly"], alpha=0.7, density=True)
    ax[1].set_title("Hour of day"); ax[1].set_xlabel("Hour")
    ax[2].scatter(n.Time / 3600, n.Amount, s=2, color=C["normal"], alpha=0.25, rasterized=True)
    ax[2].scatter(f.Time / 3600, f.Amount, s=18, color=C["anomaly"], alpha=0.9, edgecolor=C["bg"], linewidth=0.4)
    ax[2].set_yscale("log"); ax[2].set_title("Amount over time"); ax[2].set_xlabel("Hours since start")
    fig.tight_layout()
    return fig


def plot_scaling_comparison(df: pd.DataFrame):
    from sklearn.preprocessing import RobustScaler, StandardScaler
    x = df[["Amount"]]
    std = StandardScaler().fit_transform(x).ravel()
    rob = RobustScaler().fit_transform(x).ravel()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist(x.Amount, bins=80, color=C["muted"]); ax[0].set_title("Raw Amount")
    ax[1].hist(std, bins=80, color=C["alert"]); ax[1].set_title(f"StandardScaler  (max = {std.max():.0f})")
    ax[2].hist(rob, bins=80, color=C["good"]); ax[2].set_title(f"RobustScaler  (max = {rob.max():.0f})")
    for a in ax:
        a.set_yscale("log")
    fig.suptitle("Why RobustScaler: median/IQR ignore the extreme purchases that distort mean/std",
                 x=0.01, ha="left", color=C["text"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_separation(sep: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4))
    s = sep.sort_values()
    ax.barh(s.index, s.values, color=C["anomaly"], height=0.62)
    ax.set_title("Which PCA features separate fraud? (|Δ mean| / σ)"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Detection
def _pca_embed(X: pd.DataFrame, n: int):
    """Standardise + clip so a handful of extreme Amount values cannot dominate the projection."""
    from sklearn.preprocessing import StandardScaler
    Xs = np.clip(StandardScaler().fit_transform(X), -6, 6)
    pca = PCA(n_components=n, random_state=0).fit(Xs)
    return pca.transform(Xs), pca


def _groups(y, pred):
    return {"tn": (y == 0) & (pred == 0), "fp": (y == 0) & (pred == 1),
            "tp": (y == 1) & (pred == 1), "fn": (y == 1) & (pred == 0)}


def plot_pca_scatter(X: pd.DataFrame, y: np.ndarray, pred: np.ndarray, title: str, max_normal: int = 6000,
                     seed: int = 0):
    """2-D PCA map: faint normal cloud, flagged-fraud in amber, false alarms in coral, missed fraud ringed."""
    Z, pca = _pca_embed(X, 2)
    g = _groups(y, pred)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(g["tn"]); idx = rng.choice(idx, min(max_normal, idx.size), replace=False)
    fig, ax = plt.subplots(figsize=(8.2, 6))
    ax.scatter(Z[idx, 0], Z[idx, 1], s=5, color=C["normal"], alpha=0.28, label="Legitimate", rasterized=True)
    ax.scatter(Z[g["fp"], 0], Z[g["fp"], 1], s=26, color=C["alert"], alpha=0.85, label=f"False alarm ({g['fp'].sum()})",
               edgecolor=C["bg"], linewidth=0.4)
    ax.scatter(Z[g["fn"], 0], Z[g["fn"], 1], s=60, facecolor="none", edgecolor=C["alert"], linewidth=1.4,
               label=f"Missed fraud ({g['fn'].sum()})")
    ax.scatter(Z[g["tp"], 0], Z[g["tp"], 1], s=46, color=C["anomaly"], edgecolor=C["bg"], linewidth=0.6,
               label=f"Caught fraud ({g['tp'].sum()})", zorder=5)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)")
    ax.set_title(title); ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_pca_3d(X: pd.DataFrame, y: np.ndarray, pred: np.ndarray, title: str, max_normal: int = 3000, seed: int = 0):
    Z, pca = _pca_embed(X, 3)
    g = _groups(y, pred)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(g["tn"]); idx = rng.choice(idx, min(max_normal, idx.size), replace=False)
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(C["panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(C["panel"], 1)); axis._axinfo["grid"]["color"] = C["grid"]
    ax.scatter(*Z[idx].T, s=3, color=C["normal"], alpha=0.18, label="Legitimate")
    ax.scatter(*Z[g["fp"]].T, s=20, color=C["alert"], alpha=0.8, label="False alarm")
    ax.scatter(*Z[g["tp"]].T, s=38, color=C["anomaly"], label="Caught fraud", depthshade=False)
    ax.scatter(*Z[g["fn"]].T, s=46, facecolor="none", edgecolor=C["alert"], label="Missed fraud")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title(title)
    ax.legend(loc="upper left"); ax.view_init(elev=18, azim=-52)
    fig.tight_layout()
    return fig


def plot_confusion(cm: dict, title: str, ax=None):
    """``cm`` needs tn, fp, fn, tp. Cell colour encodes the share of its *true* row."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.6, 4))
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    rowshare = mat / mat.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("sp", [C["panel"], C["normal"]])
    ax.imshow(rowshare, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ok = i == j
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=17, fontweight="bold",
                    color=C["anomaly"] if (ok and i == 1) else C["text"] if ok else C["alert"])
            ax.text(j, i + 0.3, f"{rowshare[i, j]:.1%} of row", ha="center", fontsize=8.5, color=C["text"], alpha=0.7)
    ax.set_xticks([0, 1], ["Pred: normal", "Pred: fraud"]); ax.set_yticks([0, 1], ["True: normal", "True: fraud"])
    ax.grid(False); ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax.figure


def plot_confusion_pair(cms: dict[str, dict]):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.8 * len(cms), 4.2))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        plot_confusion(cm, name, ax=ax)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: np.ndarray, y: np.ndarray, threshold: float, title: str):
    fig, ax = plt.subplots(figsize=(8.2, 4))
    lo, hi = np.percentile(scores, [0.1, 99.95])
    bins = np.linspace(lo, hi, 70)
    ax.hist(scores[y == 0], bins=bins, color=C["normal"], alpha=0.85, density=True, label="Legitimate")
    ax.hist(scores[y == 1], bins=bins, color=C["anomaly"], alpha=0.8, density=True, label="Fraud")
    ax.axvline(threshold, color=C["alert"], lw=2, ls="--", label="Alert threshold")
    ax.set_yscale("log"); ax.set_xlabel("Anomaly score  →  more suspicious"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Evaluation
def plot_pr_curves(analyses: dict, y: np.ndarray, highlight: dict[str, dict] | None = None):
    fig, ax = plt.subplots(figsize=(8.4, 6))
    colors = [C["normal"], C["good"], C["anomaly"], C["alert"]]
    for (name, pr), col in zip(analyses.items(), colors):
        ax.plot(pr.recall, pr.precision, color=col, lw=2.4, label=f"{name}  (AP = {pr.average_precision:.3f})")
        b = pr.best
        ax.scatter(b["recall"], b["precision"], s=110, color=col, edgecolor="white", linewidth=1.6, zorder=6)
        ax.annotate(f"F1 {b['f1']:.2f}", (b["recall"], b["precision"]), xytext=(8, 8), textcoords="offset points",
                    color=col, fontsize=9.5, fontweight="bold")
    ax.axhline(y.mean(), color=C["muted"], ls=":", lw=1.2, label=f"Random baseline ({y.mean():.2%})")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall  (share of fraud caught)"); ax.set_ylabel("Precision  (share of alerts that are fraud)")
    ax.set_title("Precision–Recall curves  ·  dots mark the max-F1 operating point"); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_contamination_sweep(sweeps: dict[str, pd.DataFrame], recommended: dict[str, float] | None = None):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = [C["normal"], C["good"], C["anomaly"]]
    for (name, sw), col in zip(sweeps.items(), colors):
        ax[0].plot(sw.contamination, sw.recall, "-o", color=col, ms=4, label=name)
        ax[1].plot(sw.contamination, sw.precision, "-o", color=col, ms=4)
        ax[2].plot(sw.contamination, sw.f1, "-o", color=col, ms=4)
        if recommended and name in recommended:
            for a, col_name in zip(ax, ("recall", "precision", "f1")):
                row = sw.iloc[(sw.contamination - recommended[name]).abs().argmin()]
                a.scatter(row.contamination, row[col_name], s=130, facecolor="none", edgecolor="white", linewidth=1.6, zorder=6)
    for a, t in zip(ax, ("Recall", "Precision", "F1-score")):
        a.set_xscale("log"); a.set_title(t); a.set_xlabel("contamination (log)")
    ax[0].legend(); fig.tight_layout()
    return fig


def plot_runtime(models: dict) -> plt.Figure:
    names = list(models)
    fit = [m.fit_time for m in models.values()]; sc = [m.score_time for m in models.values()]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(names))
    ax.barh(y, fit, color=C["normal"], height=0.55, label="Fit")
    ax.barh(y, sc, left=fit, color=C["anomaly"], height=0.55, label="Score test set")
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i, (f, s) in enumerate(zip(fit, sc)):
        ax.text(f + s + 0.03, i, f"{f + s:.1f}s", va="center", color=C["text"], fontweight="bold")
    ax.set_xlabel("seconds"); ax.set_title("Execution time"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_banner(path: Path, seed: int = 7) -> Path:
    """Generative README banner: a quiet field of traffic with a few amber anomalies and a scan ring."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off"); fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
    x = np.concatenate([rng.normal(0.78, 0.13, 2600), rng.normal(0.9, 0.05, 500)])
    y = rng.normal(0.5, 0.2, x.size)
    ax.scatter(x, y, s=rng.uniform(1, 7, x.size), color=C["normal"], alpha=0.35, linewidths=0)
    ax2 = np.array([[0.93, 0.86], [0.97, 0.12], [0.52, 0.93], [0.985, 0.55], [0.7, 0.04]])
    for r, a in zip((0.07, 0.045, 0.02), (0.08, 0.14, 0.4)):
        ax.scatter(ax2[:, 0], ax2[:, 1], s=(r * 5000) ** 1.2 / 8, color=C["anomaly"], alpha=a, linewidths=0)
    ax.scatter(ax2[:, 0], ax2[:, 1], s=34, color=C["anomaly"], zorder=5)
    ax.text(0.015, 0.62, "SentinelPay", fontsize=50, fontweight="bold", color=C["text"], transform=ax.transAxes)
    ax.text(0.017, 0.30, "Real-time anomaly detection for financial transactions", fontsize=18,
            color=C["muted"], transform=ax.transAxes)
    ax.text(0.017, 0.12, "Isolation Forest  ·  Local Outlier Factor  ·  Rank Fusion  ·  NVIDIA GenAI analyst",
            fontsize=12.5, color=C["anomaly"], transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
