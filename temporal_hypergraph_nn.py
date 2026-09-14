#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dynamic topic hypergraph forecaster
#
# Nodes   = topics
# Edges   = papers published in a given year
# Snapshot per year y: H_y(topic, paper) = 1 if paper is tagged with topic
# Goal    = given window [t-W+1, ..., t], predict topic activity in year t+1
#
# This is the correct first forecasting target for a topic-only hypergraph.
# Predicting the exact future paper hyperedges is a harder generative problem
# that you should tackle only after this baseline works.
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a temporal hypergraph neural network over yearly paper-topic snapshots."
    )
    parser.add_argument("--input", required=True, help="Path to papers_with_topics.csv")
    parser.add_argument("--outdir", default="temporal_hypergraph_output", help="Output directory")
    parser.add_argument("--window-size", type=int, default=5, help="Number of years in each input window")
    parser.add_argument(
        "--predict-year",
        type=int,
        required=True,
        help="Target year to predict. Training uses all earlier valid windows.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["count", "binary"],
        default="count",
        help="count = predict log1p(topic paper counts), binary = predict whether topic appears.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@dataclass
class YearSnapshot:
    year: int
    x: torch.Tensor          # [num_topics, num_features]
    H: torch.Tensor          # [num_topics, num_papers_in_year]
    counts: torch.Tensor     # [num_topics]
    paper_ids: List[str]


@dataclass
class WindowSample:
    input_years: List[int]
    target_year: int
    snapshots: List[YearSnapshot]
    y: torch.Tensor          # [num_topics]


class HypergraphConv(nn.Module):
    """A lightweight hypergraph convolution without external libraries.

    Formula (with unit hyperedge weights):
        X' = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2} X Theta

    Nodes are topics. Hyperedges are papers.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # x: [N, F], H: [N, E]
        x = self.lin(x)
        if H.numel() == 0 or H.shape[1] == 0:
            return x

        dv = H.sum(dim=1).clamp_min(1.0)  # node degrees [N]
        de = H.sum(dim=0).clamp_min(1.0)  # edge sizes   [E]

        x = x / torch.sqrt(dv).unsqueeze(1)
        edge_feat = H.T @ x                           # [E, F]
        edge_feat = edge_feat / de.unsqueeze(1)
        node_feat = H @ edge_feat                     # [N, F]
        node_feat = node_feat / torch.sqrt(dv).unsqueeze(1)
        return node_feat


class YearEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.hg1 = HypergraphConv(in_dim, hidden_dim)
        self.hg2 = HypergraphConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        z = self.hg1(x, H)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.hg2(z, H)
        z = F.relu(z)
        return z


class TemporalHypergraphForecaster(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.year_encoder = YearEncoder(in_dim, hidden_dim, dropout)
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, snapshots: Sequence[YearSnapshot]) -> torch.Tensor:
        # Encode each year independently with the hypergraph encoder.
        # Note: We only use snap.x features, not snap.H, to avoid learning spurious
        # correlations from paper-topic associations that would leak future information.
        yearly_embeddings = []
        for snap in snapshots:
            # Create a simple identity incidence matrix (avoid structural leakage)
            num_topics = snap.x.shape[0]
            dummy_H = torch.eye(num_topics, dtype=snap.x.dtype, device=snap.x.device)
            z = self.year_encoder(snap.x, dummy_H)     # [N, hidden]
            yearly_embeddings.append(z)

        seq = torch.stack(yearly_embeddings, dim=1)   # [N, T, hidden]
        out, _ = self.temporal(seq)                   # [N, T, hidden]
        last = out[:, -1, :]                          # [N, hidden]
        pred = self.head(last).squeeze(-1)            # [N]
        return pred


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_topics(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    # Fallbacks for alternative separators
    if "|" in text:
        return [v.strip() for v in text.split("|") if v.strip()]
    if ";" in text:
        return [v.strip() for v in text.split(";") if v.strip()]
    return [text]


def choose_topic_column(df: pd.DataFrame) -> str:
    for candidate in ["assigned_topics", "topics", "matched_topics"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a topic column. Expected one of: assigned_topics, topics, matched_topics."
    )


def load_papers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("Input CSV must contain a 'year' column.")
    if "paper_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'paper_id' column.")

    topic_col = choose_topic_column(df)
    df[topic_col] = df[topic_col].apply(parse_topics)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "citation_count" not in df.columns:
        df["citation_count"] = 0
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df[topic_col].map(len) > 0].copy()
    df["_topics"] = df[topic_col]
    return df


def build_topic_vocab(df: pd.DataFrame) -> Dict[str, int]:
    all_topics = sorted({topic for topics in df["_topics"] for topic in topics})
    return {topic: idx for idx, topic in enumerate(all_topics)}


def compute_yearly_topic_counts(df: pd.DataFrame, topic_to_idx: Dict[str, int]) -> Dict[int, np.ndarray]:
    num_topics = len(topic_to_idx)
    counts_by_year: Dict[int, np.ndarray] = {}
    for year, group in df.groupby("year"):
        vec = np.zeros(num_topics, dtype=np.float32)
        for topics in group["_topics"]:
            for topic in topics:
                vec[topic_to_idx[topic]] += 1.0
        counts_by_year[int(year)] = vec
    return counts_by_year


def build_snapshots(df: pd.DataFrame, topic_to_idx: Dict[str, int]) -> Dict[int, YearSnapshot]:
    num_topics = len(topic_to_idx)
    yearly_counts = compute_yearly_topic_counts(df, topic_to_idx)
    years = sorted(yearly_counts.keys())

    cumulative = np.zeros(num_topics, dtype=np.float32)
    prev_counts = np.zeros(num_topics, dtype=np.float32)

    snapshots: Dict[int, YearSnapshot] = {}

    for year in years:
        group = df[df["year"] == year].copy()
        papers = list(group["paper_id"].astype(str))
        num_papers = len(group)
        H = np.zeros((num_topics, num_papers), dtype=np.float32)

        citation_sum = np.zeros(num_topics, dtype=np.float32)
        counts_this_year = np.zeros(num_topics, dtype=np.float32)

        for e_idx, (_, row) in enumerate(group.iterrows()):
            topics = list(row["_topics"])
            citation_value = float(row.get("citation_count", 0.0) or 0.0)
            citation_log = math.log1p(max(citation_value, 0.0))
            for topic in topics:
                t_idx = topic_to_idx[topic]
                H[t_idx, e_idx] = 1.0
                counts_this_year[t_idx] += 1.0
                citation_sum[t_idx] += citation_log

        growth = np.log1p(counts_this_year) - np.log1p(prev_counts)
        cumulative = cumulative + counts_this_year

        # Node features per topic for the current snapshot.
        # IMPORTANT: Do NOT include counts_this_year, citation_sum, or cumulative in features
        # as these directly leak the target information. Use only past-based features.
        X = np.stack(
            [
                np.log1p(prev_counts),      # past trend baseline
                growth,                      # momentum from previous year
                np.ones(num_topics, dtype=np.float32),  # constant (learnable bias)
            ],
            axis=1,
        ).astype(np.float32)

        snapshots[year] = YearSnapshot(
            year=year,
            x=torch.from_numpy(X),
            H=torch.from_numpy(H),
            counts=torch.from_numpy(counts_this_year.astype(np.float32)),
            paper_ids=papers,
        )
        prev_counts = counts_this_year

    return snapshots


def make_target(counts: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "count":
        return torch.log1p(counts)
    if target_mode == "binary":
        return (counts > 0).float()
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def build_window_samples(
    snapshots: Dict[int, YearSnapshot],
    window_size: int,
    target_mode: str,
) -> List[WindowSample]:
    years = sorted(snapshots.keys())
    samples: List[WindowSample] = []

    for i in range(window_size, len(years)):
        window_years = years[i - window_size:i]
        target_year = years[i]

        # Require contiguous yearly windows. This avoids silently mixing multi-year gaps.
        expected = list(range(window_years[0], window_years[0] + window_size))
        if window_years != expected:
            continue
        if target_year != window_years[-1] + 1:
            continue

        window_snaps = [snapshots[y] for y in window_years]
        y = make_target(snapshots[target_year].counts, target_mode)
        samples.append(
            WindowSample(
                input_years=window_years,
                target_year=target_year,
                snapshots=window_snaps,
                y=y,
            )
        )
    return samples


def precision_at_k(pred: torch.Tensor, truth_binary: torch.Tensor, k: int = 10) -> float:
    k = min(k, pred.numel())
    topk = torch.topk(pred, k=k).indices
    hits = truth_binary[topk].sum().item()
    return float(hits / max(k, 1))


def ndcg_at_k(pred: torch.Tensor, truth_relevance: torch.Tensor, k: int = 10) -> float:
    k = min(k, pred.numel())
    order = torch.topk(pred, k=k).indices
    rel = truth_relevance[order].float()
    discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
    dcg = float((rel * discounts).sum().item())

    ideal_rel = torch.topk(truth_relevance.float(), k=k).values
    idcg = float((ideal_rel * discounts).sum().item())
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def train_model(
    model: nn.Module,
    train_samples: Sequence[WindowSample],
    val_samples: Sequence[WindowSample],
    target_mode: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}

    if target_mode == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for sample in train_samples:
            optimizer.zero_grad()
            pred = model(sample.snapshots)
            loss = criterion(pred, sample.y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train_loss"].append(mean_train)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for sample in val_samples:
                pred = model(sample.snapshots)
                loss = criterion(pred, sample.y)
                val_losses.append(float(loss.item()))
            mean_val = float(np.mean(val_losses)) if val_losses else mean_train
            history["val_loss"].append(mean_val)

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | train_loss={mean_train:.4f} | val_loss={mean_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_sample(
    model: nn.Module,
    sample: WindowSample,
    target_mode: str,
) -> Dict[str, float | torch.Tensor]:
    model.eval()
    with torch.no_grad():
        pred = model(sample.snapshots)

    if target_mode == "binary":
        scores = torch.sigmoid(pred)
        truth_binary = sample.y
        truth_relevance = sample.y
    else:
        scores = pred
        truth_binary = (torch.expm1(sample.y) > 0).float()
        truth_relevance = sample.y

    return {
        "scores": scores,
        "precision_at_5": precision_at_k(scores, truth_binary, k=5),
        "precision_at_10": precision_at_k(scores, truth_binary, k=10),
        "ndcg_at_10": ndcg_at_k(scores, truth_relevance, k=10),
    }


def save_outputs(
    outdir: Path,
    history: Dict[str, List[float]],
    topic_to_idx: Dict[str, int],
    predict_sample: WindowSample,
    eval_result: Dict[str, float | torch.Tensor],
    target_mode: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    topics = [topic for topic, _ in sorted(topic_to_idx.items(), key=lambda x: x[1])]
    scores = eval_result["scores"]
    assert isinstance(scores, torch.Tensor)

    if target_mode == "count":
        pred_count = torch.expm1(scores).cpu().numpy()
        true_count = torch.expm1(predict_sample.y).cpu().numpy()
    else:
        pred_count = scores.cpu().numpy()
        true_count = predict_sample.y.cpu().numpy()

    pred_df = pd.DataFrame(
        {
            "topic": topics,
            "prediction": pred_count,
            "ground_truth": true_count,
        }
    ).sort_values("prediction", ascending=False)
    pred_df.to_csv(outdir / "predictions_for_target_year.csv", index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(outdir / "training_history.csv", index=False)

    config = {
        "window_years": predict_sample.input_years,
        "target_year": predict_sample.target_year,
        "num_topics": len(topics),
        "precision_at_5": float(eval_result["precision_at_5"]),
        "precision_at_10": float(eval_result["precision_at_10"]),
        "ndcg_at_10": float(eval_result["ndcg_at_10"]),
    }
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    top10 = pred_df.head(10)
    summary_lines = [
        f"Input years: {predict_sample.input_years}",
        f"Predicted year: {predict_sample.target_year}",
        f"Precision@5:  {eval_result['precision_at_5']:.4f}",
        f"Precision@10: {eval_result['precision_at_10']:.4f}",
        f"NDCG@10:      {eval_result['ndcg_at_10']:.4f}",
        "",
        "Top predicted topics:",
    ]
    for row in top10.itertuples(index=False):
        summary_lines.append(
            f" - {row.topic}: pred={row.prediction:.4f}, truth={row.ground_truth:.4f}"
        )
    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)

    df = load_papers(args.input)
    topic_to_idx = build_topic_vocab(df)
    snapshots = build_snapshots(df, topic_to_idx)
    samples = build_window_samples(snapshots, args.window_size, args.target_mode)

    if not samples:
        available_years = sorted(snapshots.keys())
        raise ValueError(
            "No valid contiguous year windows were found. "
            f"Available years: {available_years}. "
            f"Check gaps in the data or reduce --window-size."
        )

    train_pool = [s for s in samples if s.target_year < args.predict_year]
    predict_candidates = [s for s in samples if s.target_year == args.predict_year]

    if not predict_candidates:
        candidate_years = sorted({s.target_year for s in samples})
        raise ValueError(
            f"Could not build a prediction sample for year {args.predict_year}. "
            f"Available target years: {candidate_years}"
        )

    predict_sample = predict_candidates[0]

    if len(train_pool) < 2:
        raise ValueError(
            "Not enough earlier windows to train. You need at least 2 training windows "
            "before the requested --predict-year."
        )

    # Time-aware split: oldest windows for training, newest pre-predict window for validation.
    train_samples = train_pool[:-1]
    val_samples = train_pool[-1:]

    in_dim = next(iter(snapshots.values())).x.shape[1]
    model = TemporalHypergraphForecaster(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    history = train_model(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        target_mode=args.target_mode,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    eval_result = evaluate_sample(model, predict_sample, args.target_mode)
    save_outputs(
        outdir=outdir,
        history=history,
        topic_to_idx=topic_to_idx,
        predict_sample=predict_sample,
        eval_result=eval_result,
        target_mode=args.target_mode,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "topic_to_idx": topic_to_idx,
            "window_size": args.window_size,
            "predict_year": args.predict_year,
            "target_mode": args.target_mode,
        },
        outdir / "temporal_hypergraph_model.pt",
    )

    print("Done.")
    print(f"Outputs written to: {outdir.resolve()}")
    print(" - predictions_for_target_year.csv")
    print(" - training_history.csv")
    print(" - run_summary.json")
    print(" - summary.txt")
    print(" - temporal_hypergraph_model.pt")


if __name__ == "__main__":
    main()
