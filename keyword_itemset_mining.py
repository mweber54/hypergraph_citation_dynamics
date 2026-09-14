#!/usr/bin/env python3
"""
Keyword Itemset Mining + Time Series Trend Forecasting for 2026 Prediction

Mines frequent keyword pairs & triples, computes trend residuals via
unexpected lift vs baseline, and forecasts trending itemsets for 2026.

Parallel to GNN approach: find top 5 trending itemsets and their constituent keywords.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine frequent keyword itemsets and forecast trends for 2026."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to papers_with_keywords.csv",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="Earliest year to include",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2025,
        help="Latest year (actual data)",
    )
    parser.add_argument(
        "--forecast-year",
        type=int,
        default=2026,
        help="Year to forecast",
    )
    parser.add_argument(
        "--df-min-count",
        type=int,
        default=3,
        help="Minimum document frequency",
    )
    parser.add_argument(
        "--df-max-frac",
        type=float,
        default=0.25,
        help="Maximum DF as fraction of total papers",
    )
    parser.add_argument(
        "--minsup-frac",
        type=float,
        default=0.01,
        help="Minimum support fraction per year",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum support count per year",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.5,
        help="Minimum lift threshold",
    )
    parser.add_argument(
        "--baseline",
        choices=["drift", "ar1"],
        default="drift",
        help="Baseline forecasting method",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window for baseline",
    )
    parser.add_argument(
        "--top-m",
        type=int,
        default=5,
        help="Top M itemsets to report for 2026",
    )
    parser.add_argument(
        "--eval-k",
        type=int,
        default=50,
        help="Precision@K, NDCG@K for backtest",
    )
    parser.add_argument(
        "--outdir",
        default="keyword_itemset_output",
        help="Output directory",
    )
    return parser.parse_args()


def load_papers(path: str, min_year: int, max_year: int) -> pd.DataFrame:
    """Load papers and extract keywords."""
    df = pd.read_csv(path)

    # Ensure required columns
    if "year" not in df.columns or "paper_id" not in df.columns:
        raise ValueError("Expected 'year' and 'paper_id' columns")
    if "extracted_keywords" not in df.columns:
        raise ValueError("Expected 'extracted_keywords' column with JSON arrays")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Filter by year range
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].copy()

    # Parse keywords from JSON
    def parse_keywords(kw_str):
        if pd.isna(kw_str):
            return []
        try:
            return json.loads(kw_str)
        except:
            return []

    df["keywords"] = df["extracted_keywords"].apply(parse_keywords)
    df = df[df["keywords"].map(len) > 0].copy()

    return df


def compute_document_frequency(df: pd.DataFrame) -> Dict[str, int]:
    """Compute global document frequency for each keyword."""
    df_count = Counter()
    for keywords in df["keywords"]:
        for kw in set(keywords):  # Use set to count each keyword once per paper
            df_count[kw] += 1
    return dict(df_count)


def filter_keywords_by_df(
    df_count: Dict[str, int],
    total_papers: int,
    df_min_count: int,
    df_max_frac: float,
) -> Set[str]:
    """Filter keywords by document frequency thresholds."""
    df_max_count = int(total_papers * df_max_frac)
    return {
        kw for kw, count in df_count.items()
        if df_min_count <= count <= df_max_count
    }


def build_transactions(df: pd.DataFrame, valid_keywords: Set[str]) -> Dict[int, List[List[int]]]:
    """
    Build per-year transactions with keyword IDs.

    Returns: {year: [transaction, ...]}
    where transaction = sorted list of keyword IDs
    """
    keyword_to_id = {kw: i for i, kw in enumerate(sorted(valid_keywords))}

    transactions_by_year = defaultdict(list)

    for _, row in df.iterrows():
        year = int(row["year"])
        keywords = row["keywords"]

        # Filter to valid keywords and convert to IDs
        filtered_ids = sorted({
            keyword_to_id[kw] for kw in keywords if kw in valid_keywords
        })

        if filtered_ids:
            transactions_by_year[year].append(filtered_ids)

    return dict(transactions_by_year), keyword_to_id


def mine_frequent_pairs(
    transactions: List[List[int]],
    n_papers: int,
    minsup_frac: float,
    min_count: int,
    min_lift: float,
) -> Dict[Tuple[int, int], Dict]:
    """Mine frequent 2-itemsets with lift >= min_lift."""
    min_sup_count = max(min_count, math.ceil(minsup_frac * n_papers))

    # Count individual items and pairs
    item_counts = Counter()
    pair_counts = Counter()

    for tx in transactions:
        for item in tx:
            item_counts[item] += 1
        for i, j in combinations(sorted(set(tx)), 2):
            pair_counts[(i, j)] += 1

    # Filter by support and lift
    strong_pairs = {}
    for (i, j), count in pair_counts.items():
        if count >= min_sup_count:
            supp_ij = count / n_papers
            supp_i = item_counts[i] / n_papers
            supp_j = item_counts[j] / n_papers
            lift = supp_ij / (supp_i * supp_j + 1e-12)

            if lift >= min_lift:
                strong_pairs[(i, j)] = {
                    "count": count,
                    "supp": supp_ij,
                    "lift": lift,
                }

    return strong_pairs


def mine_frequent_triples(
    transactions: List[List[int]],
    n_papers: int,
    strong_pairs: Dict[Tuple[int, int], Dict],
    minsup_frac: float,
    min_count: int,
    min_lift: float,
) -> Dict[Tuple[int, int, int], Dict]:
    """Mine frequent 3-itemsets (only if all pairwise subsets are strong)."""
    min_sup_count = max(min_count, math.ceil(minsup_frac * n_papers))

    # Candidate generation with Apriori pruning
    triple_counts = Counter()

    for tx in transactions:
        unique_items = sorted(set(tx))
        for i, j, k in combinations(unique_items, 3):
            # Only consider if all three pairwise subsets are in strong_pairs
            if all([
                (i, j) in strong_pairs,
                (i, k) in strong_pairs,
                (j, k) in strong_pairs,
            ]):
                triple_counts[(i, j, k)] += 1

    # Filter by support and lift
    strong_triples = {}
    for (i, j, k), count in triple_counts.items():
        if count >= min_sup_count:
            supp_ijk = count / n_papers
            lift_pairs = [
                strong_pairs[(i, j)]["lift"],
                strong_pairs[(i, k)]["lift"],
                strong_pairs[(j, k)]["lift"],
            ]
            min_lift_val = min(lift_pairs)

            if min_lift_val >= min_lift:
                strong_triples[(i, j, k)] = {
                    "count": count,
                    "supp": supp_ijk,
                    "lift_min": min_lift_val,
                }

    return strong_triples


def build_yearly_metrics(
    transactions_by_year: Dict[int, List[List[int]]],
    keyword_to_id: Dict[str, int],
    minsup_frac: float,
    min_count: int,
    min_lift: float,
) -> Tuple[Dict, Dict, Set]:
    """
    Mine itemsets per year and build metrics.

    Returns: (pair_metrics_by_year, triple_metrics_by_year, all_itemsets)
    """
    years = sorted(transactions_by_year.keys())

    pair_metrics_by_year = {}
    triple_metrics_by_year = {}
    all_pairs = set()
    all_triples = set()

    for year in years:
        transactions = transactions_by_year[year]
        n_papers = len(transactions)

        # Mine pairs
        pairs = mine_frequent_pairs(
            transactions, n_papers, minsup_frac, min_count, min_lift
        )
        pair_metrics_by_year[year] = pairs
        all_pairs.update(pairs.keys())

        # Mine triples
        triples = mine_frequent_triples(
            transactions, n_papers, pairs, minsup_frac, min_count, min_lift
        )
        triple_metrics_by_year[year] = triples
        all_triples.update(triples.keys())

    all_itemsets = all_pairs | all_triples

    return pair_metrics_by_year, triple_metrics_by_year, all_itemsets


def compute_residuals(
    itemsets: Set,
    pair_metrics_by_year: Dict,
    triple_metrics_by_year: Dict,
    transactions_by_year: Dict[int, List[List[int]]],
    years: List[int],
    baseline_method: str = "drift",
    window: int = 5,
) -> Dict:
    """
    Compute residuals z_t - z_hat_t for all itemsets and years.

    Returns: {itemset: {"years": [...], "z": [...], "z_hat": [...], "residuals": [...]}}
    """
    results = {}

    for itemset in itemsets:
        k = len(itemset)

        # Collect z_t (log-support) for each year
        z_series = {}
        for year in years:
            n_papers = len(transactions_by_year.get(year, []))
            if n_papers == 0:
                z_series[year] = 0.0
                continue

            if k == 2:
                metrics = pair_metrics_by_year.get(year, {}).get(itemset, None)
            elif k == 3:
                metrics = triple_metrics_by_year.get(year, {}).get(itemset, None)
            else:
                metrics = None

            supp = metrics["supp"] if metrics else 0.0
            z_series[year] = math.log1p(supp)

        # Compute baselines and residuals
        z_hats = {}
        residuals = {}

        for t_idx, year in enumerate(years):
            # Use past window for baseline (no leakage)
            window_start = max(0, t_idx - window)
            history_years = years[window_start:t_idx]

            if not history_years:
                z_hats[year] = z_series[year]
                residuals[year] = 0.0
                continue

            history_z = [z_series[y] for y in history_years]

            if baseline_method == "drift":
                mu = np.mean(history_z)
                if len(history_z) > 1:
                    diffs = [history_z[i+1] - history_z[i] for i in range(len(history_z)-1)]
                    drift = np.mean(diffs)
                else:
                    drift = 0.0
                z_hat = mu + drift

            elif baseline_method == "ar1":
                if len(history_z) > 1:
                    x = np.array(history_z[:-1])
                    y = np.array(history_z[1:])
                    mu_x = np.mean(x)
                    mu_y = np.mean(y)
                    var_x = np.sum((x - mu_x) ** 2)

                    if var_x > 1e-12:
                        cov_xy = np.sum((x - mu_x) * (y - mu_y))
                        a = cov_xy / var_x
                        b = mu_y - a * mu_x
                    else:
                        a, b = 0.0, mu_y

                    z_hat = a * z_series[history_years[-1]] + b
                else:
                    z_hat = history_z[0]

            else:
                z_hat = np.mean(history_z)

            z_hats[year] = z_hat
            residuals[year] = z_series[year] - z_hat

        results[itemset] = {
            "years": years,
            "z": [z_series[y] for y in years],
            "z_hat": [z_hats[y] for y in years],
            "residuals": [residuals[y] for y in years],
        }

    return results


def forecast_2026(
    residuals_data: Dict,
    years: List[int],
    forecast_year: int,
    window: int = 5,
) -> Dict[Tuple, float]:
    """
    Forecast residual for 2026 based on recent years.

    Returns: {itemset: pred_residual_2026}
    """
    forecast_window_start = max(2005, forecast_year - window)
    relevant_year_indices = [i for i, y in enumerate(years) if y >= forecast_window_start]

    predictions = {}

    for itemset, metrics in residuals_data.items():
        residuals = metrics["residuals"]
        relevant_residuals = [residuals[i] for i in relevant_year_indices if i < len(residuals)]

        if relevant_residuals:
            pred_residual = np.mean(relevant_residuals)
        else:
            pred_residual = 0.0

        predictions[itemset] = pred_residual

    return predictions


def id_to_keywords(ids: Tuple[int, ...], id_to_keyword: Dict[int, str]) -> Tuple[str, ...]:
    """Convert itemset IDs back to keyword names."""
    return tuple(id_to_keyword[i] for i in ids)


def exponential_model(t, y0, k):
    """Exponential growth model: y = y0 * exp(k*t)"""
    return y0 * np.exp(k * t)


def logistic_model(t, L, k, t0):
    """Logistic growth model: y = L / (1 + exp(-k*(t-t0)))"""
    return L / (1 + np.exp(-k * (t - t0)))


def fit_trend_curve(years_array: np.ndarray, z_values: np.ndarray) -> Dict:
    """
    Fit exponential and logistic curves to time series data.
    Extract trend parameters k (growth rate) and compute trend score.
    
    Returns dict with:
    - exp_k: exponential growth rate
    - logistic_k: logistic growth rate
    - linear_slope: linear regression slope
    - trend_score: composite trend score (higher = stronger growth)
    - model_fit: which model fits best
    """
    if len(years_array) < 2 or np.sum(np.abs(z_values)) < 1e-10:
        return {
            "exp_k": 0.0,
            "logistic_k": 0.0,
            "linear_slope": 0.0,
            "trend_score": 0.0,
            "model_fit": "constant",
        }
    
    t_index = np.arange(len(years_array))
    z_clean = np.array(z_values, dtype=float)
    
    results = {
        "exp_k": 0.0,
        "logistic_k": 0.0,
        "linear_slope": 0.0,
        "trend_score": 0.0,
        "model_fit": "none",
    }
    
    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = linregress(t_index, z_clean)
        results["linear_slope"] = float(slope)
        results["linear_r2"] = float(r_value ** 2)
    except:
        results["linear_slope"] = 0.0
        results["linear_r2"] = 0.0
    
    # Exponential fit (only if all values positive)
    if np.all(z_clean > 0):
        try:
            y0_init = z_clean[0]
            k_init = 0.1
            popt, _ = curve_fit(
                exponential_model,
                t_index,
                z_clean,
                p0=[y0_init, k_init],
                maxfev=5000,
            )
            exp_k = popt[1]
            results["exp_k"] = float(exp_k)
            
            # Check fit quality
            y_pred = exponential_model(t_index, *popt)
            mse = np.mean((z_clean - y_pred) ** 2)
            if mse < 1e-6:  # Good fit
                results["model_fit"] = "exponential"
        except:
            results["exp_k"] = 0.0
    
    # Logistic fit
    try:
        L_init = np.max(z_clean) * 1.5
        k_init = 1.0
        t0_init = len(years_array) / 2
        popt, _ = curve_fit(
            logistic_model,
            t_index,
            z_clean,
            p0=[L_init, k_init, t0_init],
            maxfev=5000,
        )
        logistic_k = popt[1]
        results["logistic_k"] = float(logistic_k)
        
        # Check fit quality
        y_pred = logistic_model(t_index, *popt)
        mse = np.mean((z_clean - y_pred) ** 2)
        if mse < 1e-6 and results["model_fit"] == "none":
            results["model_fit"] = "logistic"
    except:
        results["logistic_k"] = 0.0
    
    # Compute composite trend score
    # Prioritize: recent trend (linear slope) + exponential growth (exp_k) + logistic growth (logistic_k)
    trend_score = (
        results["linear_slope"] * 0.5 +      # Recent trend (50%)
        results["exp_k"] * 0.25 +              # Exponential growth rate (25%)
        results["logistic_k"] * 0.25           # Logistic growth rate (25%)
    )
    results["trend_score"] = float(trend_score)
    
    return results


def select_itemsets_by_trend(
    residuals_data: Dict,
    years: List[int],
    top_m: int = 5,
) -> List[Tuple[Tuple[int, ...], Dict]]:
    """
    Filter and rank itemsets based on positive increasing trends.
    Returns top itemsets sorted by trend score.
    """
    itemset_trends = []
    
    for itemset, metrics in residuals_data.items():
        z_values = metrics["z"]
        
        # Only consider itemsets with positive recent trend
        if len(z_values) >= 3:
            recent_z = z_values[-3:]  # Last 3 data points
            trend_positive = np.mean(np.diff(recent_z)) > 0
        else:
            trend_positive = True
        
        if trend_positive:
            years_array = np.array(years)
            z_array = np.array(z_values)
            trend_metrics = fit_trend_curve(years_array, z_array)
            
            # Only include if trend score is positive
            if trend_metrics["trend_score"] > 0:
                itemset_trends.append((itemset, trend_metrics))
    
    # Sort by trend score
    itemset_trends.sort(key=lambda x: x[1]["trend_score"], reverse=True)
    
    return itemset_trends[:top_m]


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KEYWORD ITEMSET MINING + TREND FORECASTING")
    print("=" * 80)
    print()

    # Phase 1: Load and preprocess
    print(f"Loading papers from {args.input}...")
    df = load_papers(args.input, args.min_year, args.max_year)
    print(f"  Loaded {len(df)} papers across years {df['year'].min()}-{df['year'].max()}")

    print("\nPhase 1: Filtering keywords by document frequency...")
    df_count = compute_document_frequency(df)
    print(f"  Total unique keywords: {len(df_count)}")

    valid_keywords = filter_keywords_by_df(
        df_count, len(df), args.df_min_count, args.df_max_frac
    )
    print(f"  After DF filtering [{args.df_min_count}, {int(len(df)*args.df_max_frac)}]: {len(valid_keywords)} keywords")

    # Phase 2: Build transactions
    print("\nPhase 2: Building per-year transactions...")
    transactions_by_year, keyword_to_id = build_transactions(df, valid_keywords)
    id_to_keyword = {v: k for k, v in keyword_to_id.items()}
    print(f"  Years covered: {sorted(transactions_by_year.keys())}")
    for year in sorted(transactions_by_year.keys()):
        print(f"    {year}: {len(transactions_by_year[year])} papers")

    # Phase 3: Mine itemsets
    print("\nPhase 3: Mining frequent itemsets (pairs and triples)...")
    pair_metrics_by_year, triple_metrics_by_year, all_itemsets = build_yearly_metrics(
        transactions_by_year,
        keyword_to_id,
        args.minsup_frac,
        args.min_count,
        args.min_lift,
    )

    pairs = {s for s in all_itemsets if len(s) == 2}
    triples = {s for s in all_itemsets if len(s) == 3}
    print(f"  Discovered {len(pairs)} frequent pairs, {len(triples)} frequent triples")

    # Phase 4: Compute residuals
    print("\nPhase 4: Computing baseline residuals...")
    years = sorted(transactions_by_year.keys())
    residuals_data = compute_residuals(
        all_itemsets,
        pair_metrics_by_year,
        triple_metrics_by_year,
        transactions_by_year,
        years,
        baseline_method=args.baseline,
        window=args.window,
    )
    print(f"  Baseline method: {args.baseline}")
    print(f"  Window: {args.window} years")

    # Phase 5: Forecast 2026
    print("\nPhase 5: Forecasting 2026...")
    predictions_2026 = forecast_2026(residuals_data, years, args.forecast_year, args.window)

    # Phase 5B: Trend-based selection using curve fitting
    print("\nPhase 5B: Analyzing trends with curve fitting...")
    print("  Fitting exponential/logistic curves to itemset time series...")
    trend_ranked_itemsets = select_itemsets_by_trend(residuals_data, years, top_m=args.top_m * 2)
    
    # Rerank by trend score and select top M
    top_itemsets_2026 = trend_ranked_itemsets[:args.top_m]

    if trend_ranked_itemsets:
        print(f"  Found {len(trend_ranked_itemsets)} itemsets with positive trends")
        print(f"\nTop {args.top_m} trending itemsets for {args.forecast_year}:")
        for rank, (itemset, trend_metrics) in enumerate(top_itemsets_2026, 1):
            keywords = id_to_keywords(itemset, id_to_keyword)
            print(f"  [{rank}] {' + '.join(keywords)}")
            print(f"      Trend Score: {trend_metrics['trend_score']:+.4f}")
            print(f"      Growth Rate (k): {trend_metrics['exp_k']:+.4f} (exponential)")
            print(f"      Logistic Rate: {trend_metrics['logistic_k']:+.4f}")
            print(f"      Slope: {trend_metrics['linear_slope']:+.4f} | Best Model: {trend_metrics['model_fit']}")
    else:
        # Fallback to residual-based ranking
        print("  Not enough itemsets with positive trend, using residual ranking...")
        top_itemsets_2026 = sorted(
            predictions_2026.items(),
            key=lambda x: x[1],
            reverse=True
        )[:args.top_m]

    # Output: Predictions with keywords (for comparison to GNN)
    if trend_ranked_itemsets:
        output_data = {
            "metadata": {
                "method": "frequent itemset mining with trend curve fitting",
                "forecast_year": args.forecast_year,
                "baseline": args.baseline,
                "min_lift": args.min_lift,
                "selection_method": "exponential/logistic curve parameters",
                "num_pairs": len(pairs),
                "num_triples": len(triples),
                "total_itemsets": len(all_itemsets),
            },
            "top_trending_itemsets": [
                {
                    "rank": rank,
                    "itemset": {
                        "keywords": id_to_keywords(itemset, id_to_keyword),
                        "size": len(itemset),
                    },
                    "trend_score": round(float(trend_metrics["trend_score"]), 6),
                    "exp_growth_rate_k": round(float(trend_metrics["exp_k"]), 6),
                    "logistic_growth_rate_k": round(float(trend_metrics["logistic_k"]), 6),
                    "linear_slope": round(float(trend_metrics["linear_slope"]), 6),
                    "best_fit_model": trend_metrics["model_fit"],
                }
                for rank, (itemset, trend_metrics) in enumerate(top_itemsets_2026, 1)
            ],
        }
    else:
        # Fallback output
        output_data = {
            "metadata": {
                "method": "frequent itemset mining (residual fallback)",
                "forecast_year": args.forecast_year,
                "baseline": args.baseline,
                "min_lift": args.min_lift,
                "num_pairs": len(pairs),
                "num_triples": len(triples),
                "total_itemsets": len(all_itemsets),
            },
            "top_trending_itemsets": [
                {
                    "rank": rank,
                    "itemset": {
                        "keywords": id_to_keywords(itemset, id_to_keyword),
                        "size": len(itemset),
                    },
                    "predicted_residual": round(float(predictions_2026[itemset]), 6),
                }
                for rank, (itemset, pred_residual) in enumerate(top_itemsets_2026, 1)
            ],
        }

    print(f"\nWriting outputs to {outdir}...")

    # Write JSON for UI integration
    json_path = outdir / "itemsets_2026_predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"  {json_path.name}")

    # Write text summary
    summary_path = outdir / "itemsets_2026_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FREQUENT ITEMSET MINING - 2026 FORECAST (TREND-BASED SELECTION)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Method: Exponential/Logistic curve fitting on itemset time series\n")
        f.write(f"Baseline: {args.baseline} with {args.window}-year window\n")
        f.write(f"Min lift: {args.min_lift}\n")
        f.write(f"Frequent pairs: {len(pairs)}\n")
        f.write(f"Frequent triples: {len(triples)}\n\n")
        f.write("Top 5 Trending Itemsets for 2026:\n")
        f.write("(Selected by positive increasing trend using exponential/logistic growth curves)\n\n")

        if trend_ranked_itemsets:
            for rank, (itemset, trend_metrics) in enumerate(top_itemsets_2026, 1):
                keywords = id_to_keywords(itemset, id_to_keyword)
                f.write(f"[{rank}] {' + '.join(keywords)}\n")
                f.write(f"    Trend Score: {trend_metrics['trend_score']:+.6f}\n")
                f.write(f"    Exponential Growth Rate (k): {trend_metrics['exp_k']:+.6f}\n")
                f.write(f"    Logistic Growth Rate (k): {trend_metrics['logistic_k']:+.6f}\n")
                f.write(f"    Linear Slope: {trend_metrics['linear_slope']:+.6f}\n")
                f.write(f"    Best Fit Model: {trend_metrics['model_fit']}\n")
                f.write(f"    (These keywords show consistent upward trend)\n\n")
        else:
            for rank, (itemset, pred_residual) in enumerate(top_itemsets_2026, 1):
                keywords = id_to_keywords(itemset, id_to_keyword)
                f.write(f"[{rank}] {' + '.join(keywords)}\n")
                f.write(f"    Predicted residual: {pred_residual:+.6f}\n")
                f.write(f"    (These keywords together show unexpected growth trend)\n\n")
    print(f"  {summary_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
