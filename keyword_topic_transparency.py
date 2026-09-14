#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
except Exception as e:
    raise RuntimeError(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from e


GENERIC_CS_TERMS = {
    "paper", "approach", "method", "methods", "model", "models", "system", "systems",
    "framework", "frameworks", "study", "studies", "analysis", "results", "using", "based",
    "toward", "towards", "via", "new", "novel", "efficient", "robust", "improved",
    "computer", "science", "research", "problem", "problems", "application", "applications",
    "task", "tasks", "dataset", "datasets", "algorithm", "algorithms", "learning",
    # Publication/source noise
    "arxiv", "org", "ieee", "access", "acm", "springer", "journal", "conference", 
    "proceedings", "thing", "things", "review", "transaction", "symposium", "chapter",
    "meeting", "survey", "international", "transactions",
    # Conference names (should be filtered out)
    "aacl", "icml", "nips", "iccv", "cvpr", "aaai", "ijcai", "kdd", "sigmod",
    # Generic action words
    "scale", "use", "does", "using", "as", "well", "can", "may", "should",
    "do", "is", "are", "be", "have", "has", "been", "given", "made",
    # Generic descriptors
    "reason", "note", "why", "how", "what", "when", "where", "which",
    # Generic nouns that don't add semantic meaning
    "vol", "pp", "pages", "page", "edition", "press", "publishing", "publisher",
    "data", "information", "value", "set", "type", "part", "way", "state",
}

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union(GENERIC_CS_TERMS)
TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract paper keywords and compute transparent keyword influence scores for each topic over a temporal window."
    )
    p.add_argument("--input", required=True, help="Path to papers_with_topics.csv")
    p.add_argument("--outdir", default="keyword_topic_transparency_output", help="Output directory")
    p.add_argument("--window-start", type=int, required=True, help="First input year in the explanation window")
    p.add_argument("--window-end", type=int, required=True, help="Last input year in the explanation window")
    p.add_argument("--target-year", type=int, required=True, help="Forecast target year t+1")
    p.add_argument("--top-keywords-per-paper", type=int, default=5, help="How many keywords to keep per paper")
    p.add_argument("--top-keywords-per-topic", type=int, default=15, help="How many keywords to save per topic")
    p.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for keyword candidates")
    p.add_argument("--max-features", type=int, default=20000, help="Maximum ngram features for TF-IDF vocabulary")
    p.add_argument("--ngrams", default="1,2,3", help="Comma-separated ngram sizes to consider, e.g. 1,2,3")
    return p.parse_args()


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
    if "|" in text:
        return [v.strip() for v in text.split("|") if v.strip()]
    if ";" in text:
        return [v.strip() for v in text.split(";") if v.strip()]
    return [text]


def choose_topic_column(df: pd.DataFrame) -> str:
    for candidate in ["assigned_topics", "topics", "matched_topics"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a topic column. Expected one of: assigned_topics, topics, matched_topics.")


def choose_text_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["title", "abstract", "summary", "keywords", "fields_of_study", "venue"]:
        if c in df.columns:
            cols.append(c)
    if not cols:
        raise ValueError("Need at least one text column such as title or abstract.")
    return cols


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_keyword(keyword: str) -> str:
    """
    Normalize keyword variants to canonical forms.
    Handles:
    - Singular/plural consolidation
    - Multi-word phrase consolidation
    - Suffix removal (scalable, probabilistic, based, etc.)
    - Conference/journal name removal
    - Generic phrase consolidation
    """
    keyword = keyword.strip().lower()
    
    # Fix double plurals/typos first (autoencoderss -> autoencoders)
    keyword = re.sub(r'\b(\w+)ss\b', r'\1s', keyword)
    
    # REMOVE conference/journal names entirely - don't include them
    conferences = {
        'aacl', 'icml', 'nips', 'iccv', 'cvpr', 'aaai', 'ijcai', 'kdd', 'sigmod',
        'acl', 'emnlp', 'naacl', 'coling', 'infocom', 'sigcomm', 'osdi', 'sosp',
    }
    tokens = keyword.split()
    if any(conf in tokens for conf in conferences):
        # Conference name detected - remove from keyword
        tokens = [t for t in tokens if t not in conferences]
        if not tokens:
            return ""  # Return empty if only conference name
        keyword = " ".join(tokens)
    
    # MULTI-WORD CONSOLIDATIONS (handle phrases with different lengths/variations)
    # Convert dict to list with tuples for consistent ordering
    multi_word_rules = [
        # Query optimization consolidations (all query-related → query optimization)
        (r'\bquery optimization\b', 'query optimization'),
        (r'\bquery processing\b', 'query optimization'),
        (r'\bSQL query\b', 'query optimization'),
        (r'\bsql optimization\b', 'query optimization'),
        (r'\bqueries\b', 'query optimization'),
        (r'\bquery\b', 'query optimization'),  # Last - most general
        
        # Remote sensing consolidations (all remote-related → remote sensing)
        (r'\bremote sensing\b', 'remote sensing'),
        (r'\bremote image\b', 'remote sensing'),
        (r'\bremote imagery\b', 'remote sensing'),
        (r'\bsatellite remote sensing\b', 'remote sensing'),
        (r'\bsatellite imaging\b', 'remote sensing'),
        # Note: Be cautious with bare "remote" - could apply general "remote" → "remote sensing"
        
        # Large language model → llm (most specific first)
        (r'\blarge language model\b', 'llm'),
        (r'\blarge language\b', 'llm'),
        (r'\bllms?\b', 'llm'),
        
        # Cloud computing consolidations
        (r'\bcloud computing international\b', 'cloud computing'),
        
        # Distributed systems consolidations  
        (r'\bdistributed international\b', 'distributed'),
        (r'\bdistributed systems\b', 'distributed systems'),
        
        # Diffusion model consolidations
        (r'\bdenoising diffusion probabilistic\b', 'denoising diffusion'),
        (r'\btext to image diffusion\b', 'text-to-image diffusion'),
        (r'\btext-to-image diffusion\b', 'text-to-image diffusion'),
        (r'\btext to image\b', 'diffusion'),  # Single text-to-image becomes diffusion
        (r'\btext-to-image\b', 'diffusion'),  
        
        # Autoencoder consolidations - remove scalable suffix
        (r'\bautoencoders?\s+scalable\b', 'autoencoders'),
        (r'\bvariational autoencoders?\b', 'autoencoders'),
        (r'\bautoencoders?\b', 'autoencoders'),
        
        # Retrieval augmented generation
        (r'\bretrievel-augmented generation\b', 'retrieval-augmented generation'),
        (r'\bretrieval-augmented generation\b', 'retrieval-augmented generation'),
        (r'\bretrieval.?augmented\b', 'retrieval-augmented generation'),
        
        # Fairness/accountability/transparency - consolidate longer versions
        (r'\bfairness accountability transparency\b', 'fairness accountability'),
        (r'\bfairness accountability\b', 'fairness accountability'),
        
        # Vector machines → SVM
        (r'\bsupport vector machines?\b', 'support vector machine'),
        (r'\bvector machines?\b', 'support vector machine'),
        
        # Software engineering consolidations (all software-related → software engineering)
        (r'\bsoftware engineering\b', 'software engineering'),
        (r'\bsoftware\b', 'software engineering'),
        
        # Interpretability/Fairness/Explainability consolidations (AI safety concepts)
        (r'\bexplainability\b', 'interpretability'),
        (r'\bfairness\b', 'interpretability'),
        (r'\btransparency\b', 'interpretability'),
        (r'\bexplainable\b', 'interpretability'),
        (r'\bfair machine learning\b', 'interpretability'),
        (r'\bfairness accountability\b', 'interpretability'),
        
        # Reinforcement learning consolidations (all RL variants → reinforcement learning)
        (r'\breinforcement learning\b', 'reinforcement learning'),
        (r'\bq-learning\b', 'reinforcement learning'),
        (r'\bpolicy gradient\b', 'reinforcement learning'),
        (r'\bpolicy learning\b', 'reinforcement learning'),
        (r'\bdeep reinforcement learning\b', 'reinforcement learning'),
        (r'\bdrl\b', 'reinforcement learning'),
        (r'\b(?:^|\s)rl(?:\s|$)\b', 'reinforcement learning'),
        (r'\bmarkov decision process\b', 'reinforcement learning'),
        (r'\bmdp\b', 'reinforcement learning'),
        
        # Neural architecture consolidations (CNN, RNN, LSTM → neural architecture)
        (r'\bconvolutional neural network\b', 'convolutional neural network'),
        (r'\bcnn\b', 'convolutional neural network'),
        (r'\brecurrent neural network\b', 'recurrent neural network'),
        (r'\brnn\b', 'recurrent neural network'),
        (r'\blong short-term memory\b', 'lstm'),
        (r'\blstm\b', 'lstm'),
        (r'\bgru\b', 'gated recurrent unit'),
        (r'\bgated recurrent unit\b', 'gated recurrent unit'),
        
        # Attention mechanisms consolidations (attention, transformer, etc. → attention mechanism)
        (r'\bself-attention\b', 'attention mechanism'),
        (r'\bself attention\b', 'attention mechanism'),
        (r'\bmulti-head attention\b', 'attention mechanism'),
        (r'\bmulti head attention\b', 'attention mechanism'),
        (r'\btransformer\b', 'attention mechanism'),
        (r'\battention mechanism\b', 'attention mechanism'),
        (r'\battention\b', 'attention mechanism'),
        (r'\bbert\b', 'attention mechanism'),
        (r'\bgpt\b', 'attention mechanism'),
        (r'\bvision transformer\b', 'attention mechanism'),
        (r'\bvit\b', 'attention mechanism'),
        
        # Reasoning variants
        (r'\breasoning language\b', 'reasoning'),
        (r'\breason language\b', 'reasoning'),
        (r'\bllm reasoning\b', 'llm reasoning'),
        
        # Generative models
        (r'\bgenerative adversarial networks?\b', 'generative adversarial network'),
        (r'\bgenerative models?\b', 'generative model'),
        
        # Neural networks variants
        (r'\bneural networks?\b', 'neural network'),
        
        # Ensemble methods
        (r'\bensemble methods?\b', 'ensemble method'),
    ]
    
    # Apply multi-word consolidation rules - apply all rules without breaking
    for pattern, replacement in multi_word_rules:
        keyword = re.sub(pattern, replacement, keyword, flags=re.IGNORECASE)
    
    # SINGULAR/PLURAL consolidations (general)
    singular_rules = [
        (r'\bmodels?\b', 'model'),
        (r'\bnetworks?\b', 'network'),
        (r'\balgorithms?\b', 'algorithm'),
        (r'\bmethods?\b', 'method'),
        (r'\bsystems?\b', 'system'),
        (r'\bdatasets?\b', 'dataset'),
        (r'\bfeatures?\b', 'feature'),
        (r'\blayers?\b', 'layer'),
        (r'\bneurons?\b', 'neuron'),
        (r'\bfilters?\b', 'filter'),
        (r'\bkernels?\b', 'kernel'),
        (r'\bfunctions?\b', 'function'),
        (r'\bclassifiers?\b', 'classifier'),
        (r'\bregressors?\b', 'regressor'),
    ]
    
    for pattern, replacement in singular_rules:
        keyword = re.sub(pattern, replacement, keyword)
    
    # REMOVE COMMON SUFFIXES that create duplicates
    # Remove: -scalable, -based, -probabilistic, -accelerated, -enhanced, etc.
    keywords_to_remove_suffixes = ['scalable', 'based', 'probabilistic', 'enhanced', 
                                    'accelerated', 'efficient', 'large', 'small', 'improved']
    tokens = keyword.split()
    filtered_tokens = []
    for token in tokens:
        keep = True
        for suffix in keywords_to_remove_suffixes:
            if token.endswith(suffix) and token != suffix:
                # Only remove if it's truly a suffix (the base is there too elsewhere)
                base = token[:-len(suffix)].rstrip('-_')
                if len(base) > 2:  # Ensure we're not removing the whole word
                    keep = False
                    break
        if keep:
            filtered_tokens.append(token)
    
    keyword = " ".join(filtered_tokens) if filtered_tokens else keyword
    
    # Remove duplicate consecutive words (e.g., "diffusion diffusion" -> "diffusion")
    words = keyword.split()
    deduped_words = []
    for i, word in enumerate(words):
        if i == 0 or word != words[i-1]:
            deduped_words.append(word)
    keyword = " ".join(deduped_words)
    
    # Clean up extra spaces
    keyword = re.sub(r'\s+', ' ', keyword).strip()
    
    # Final check: remove if it becomes empty or is just single generic stopword
    if not keyword or keyword in CUSTOM_STOPWORDS:
        return ""
    
    return keyword


def build_document_text(row: pd.Series, text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        val = row.get(col, "")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        parts.append(str(val))
    return normalize_text(" ".join(parts))


def load_papers(path: str) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(path)
    if "paper_id" not in df.columns or "year" not in df.columns:
        raise ValueError("Input must contain paper_id and year columns.")
    topic_col = choose_topic_column(df)
    text_columns = choose_text_columns(df)

    df[topic_col] = df[topic_col].apply(parse_topics)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citation_count"] = pd.to_numeric(df.get("citation_count", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df[topic_col].map(len) > 0].copy()
    df["_topics"] = df[topic_col]
    df["_doc_text"] = df.apply(lambda row: build_document_text(row, text_columns), axis=1)
    return df, topic_col, text_columns


def valid_phrase(phrase: str) -> bool:
    phrase = phrase.strip().lower()
    if not phrase:
        return False
    tokens = phrase.split()
    if len(tokens) == 0:
        return False
    if all(tok in CUSTOM_STOPWORDS for tok in tokens):
        return False
    if any(len(tok) <= 1 for tok in tokens):
        return False
    if len(tokens) == 1 and (tokens[0].isdigit() or tokens[0] in CUSTOM_STOPWORDS):
        return False
    
    # Reject phrases that look like URLs or source citations
    if any(noise in phrase for noise in ["arxiv", "org", "doi", "http", "aacl", "acl"]):
        return False
    
    # Reject if contains mostly stopwords
    stopword_ratio = sum(1 for tok in tokens if tok in CUSTOM_STOPWORDS) / len(tokens)
    if stopword_ratio > 0.4:  # Strict cutoff
        return False
    
    # Reject very short numeric-heavy phrases
    digit_ratio = sum(1 for tok in tokens if any(c.isdigit() for c in tok)) / len(tokens)
    if digit_ratio > 0.5 and len(tokens) <= 2:
        return False
    
    # Additional: reject if > 60% of phrase is in CUSTOM_STOPWORDS by character count
    stopword_chars = sum(len(tok) for tok in tokens if tok in CUSTOM_STOPWORDS)
    total_chars = sum(len(tok) for tok in tokens)
    if total_chars > 0 and stopword_chars / total_chars > 0.5:
        return False
    
    return True


def build_keyword_matrix(
    docs: List[str],
    min_df: int,
    max_features: int,
    ngram_sizes: Tuple[int, int],
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_RE.pattern,
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=ngram_sizes,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    X = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    keep_mask = np.array([valid_phrase(term) for term in vocab], dtype=bool)
    X = X[:, keep_mask]
    vocab = vocab[keep_mask]
    return X, vocab


def top_keywords_for_paper(X, vocab: np.ndarray, top_k: int) -> List[List[Tuple[str, float]]]:
    results: List[List[Tuple[str, float]]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue
        order = np.argsort(row.data)[::-1][:top_k]
        # Apply normalization to keywords
        keywords = [(normalize_keyword(str(vocab[row.indices[j]])), float(row.data[j])) for j in order]
        results.append(keywords)
    return results


def year_weight(year: int, window_start: int, window_end: int) -> float:
    # Recency weighting inside the input window.
    # Oldest year gets 1.0, newest gets 2.0.
    if window_end <= window_start:
        return 1.0
    return 1.0 + (year - window_start) / (window_end - window_start)


def compute_keyword_topic_influence(
    df: pd.DataFrame,
    paper_keywords: List[List[Tuple[str, float]]],
    window_start: int,
    window_end: int,
    target_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if window_end < window_start:
        raise ValueError("window_end must be >= window_start")
    if target_year <= window_end:
        raise ValueError("target_year should be after the input window")

    window_df = df[(df["year"] >= window_start) & (df["year"] <= window_end)].copy()
    target_df = df[df["year"] == target_year].copy()
    if window_df.empty:
        raise ValueError("No papers fall inside the requested input window.")

    window_idx = set(window_df.index.tolist())
    target_idx = set(target_df.index.tolist())

    # Global keyword support for specificity.
    global_window_keyword_weight = Counter()
    global_target_keyword_weight = Counter()

    topic_keyword_window = defaultdict(Counter)
    topic_keyword_target = defaultdict(Counter)
    topic_paper_window = Counter()
    topic_paper_target = Counter()

    for idx, row in df.iterrows():
        kws = paper_keywords[idx]
        topics = row["_topics"]
        y = int(row["year"])

        if idx in window_idx:
            yw = year_weight(y, window_start, window_end)
            citation_w = math.log1p(max(float(row.get("citation_count", 0.0) or 0.0), 0.0)) + 1.0
            for kw, kw_score in kws:
                contribution = kw_score * yw * citation_w
                global_window_keyword_weight[kw] += contribution
                for topic in topics:
                    topic_keyword_window[topic][kw] += contribution
            for topic in topics:
                topic_paper_window[topic] += 1

        if idx in target_idx:
            citation_w = math.log1p(max(float(row.get("citation_count", 0.0) or 0.0), 0.0)) + 1.0
            for kw, kw_score in kws:
                contribution = kw_score * citation_w
                global_target_keyword_weight[kw] += contribution
                for topic in topics:
                    topic_keyword_target[topic][kw] += contribution
            for topic in topics:
                topic_paper_target[topic] += 1

    # Consolidate global keywords that normalized to same form
    global_window_consolidated = Counter()
    for kw, weight in global_window_keyword_weight.items():
        global_window_consolidated[kw] += weight
    
    global_target_consolidated = Counter()
    for kw, weight in global_target_keyword_weight.items():
        global_target_consolidated[kw] += weight

    all_topics = sorted(set(topic_keyword_window) | set(topic_keyword_target))
    rows = []

    for topic in all_topics:
        # Consolidate duplicate keywords that were normalized to same form
        window_keywords_consolidated = Counter()
        for kw, weight in topic_keyword_window[topic].items():
            window_keywords_consolidated[kw] += weight
        
        target_keywords_consolidated = Counter()
        for kw, weight in topic_keyword_target[topic].items():
            target_keywords_consolidated[kw] += weight
        
        window_total = sum(window_keywords_consolidated.values()) + 1e-9
        global_total = sum(global_window_consolidated.values()) + 1e-9
        keywords = set(window_keywords_consolidated.keys()) | set(target_keywords_consolidated.keys())

        for kw in keywords:
            win_w = window_keywords_consolidated[kw]
            tgt_w = target_keywords_consolidated[kw]
            global_w = global_window_consolidated.get(kw, 0)
            global_tgt = global_target_consolidated.get(kw, 0)

            topic_prob = win_w / window_total
            global_prob = global_w / global_total
            specificity = math.log((topic_prob + 1e-9) / (global_prob + 1e-9))

            # Trend: compare target-year presence vs window presence.
            trend_ratio = math.log1p(tgt_w) - math.log1p(win_w)

            # A direct relevance signal within the topic window.
            raw_strength = math.log1p(win_w)

            # Does the keyword continue into the target year for this topic?
            continuity = math.log1p(tgt_w)

            rows.append(
                {
                    "topic": topic,
                    "keyword": kw,
                    "window_weight": win_w,
                    "target_weight": tgt_w,
                    "global_window_weight": global_w,
                    "global_target_weight": global_tgt,
                    "topic_specificity": specificity,
                    "trend_ratio": trend_ratio,
                    "raw_strength": raw_strength,
                    "continuity": continuity,
                    "topic_window_papers": int(topic_paper_window[topic]),
                    "topic_target_papers": int(topic_paper_target[topic]),
                }
            )

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise ValueError("No keyword-topic scores could be computed. Try lowering min_df or increasing data coverage.")

    # Normalize within each topic so large topics do not dominate purely by size.
    def zscore_col(s: pd.Series) -> pd.Series:
        std = float(s.std(ddof=0))
        if std < 1e-12:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - float(s.mean())) / std

    scores["z_strength"] = scores.groupby("topic")["raw_strength"].transform(zscore_col)
    scores["z_specificity"] = scores.groupby("topic")["topic_specificity"].transform(zscore_col)
    scores["z_continuity"] = scores.groupby("topic")["continuity"].transform(zscore_col)
    scores["z_trend"] = scores.groupby("topic")["trend_ratio"].transform(zscore_col)

    # Influence score: mainly relevance + specificity, then continuity/trend.
    scores["influence_score"] = (
        0.40 * scores["z_strength"]
        + 0.30 * scores["z_specificity"]
        + 0.20 * scores["z_continuity"]
        + 0.10 * scores["z_trend"]
    )

    scores = scores.sort_values(["topic", "influence_score"], ascending=[True, False]).reset_index(drop=True)

    topic_summary_rows = []
    for topic, g in scores.groupby("topic", sort=False):
        top = g.head(15)
        topic_summary_rows.append(
            {
                "topic": topic,
                "num_window_papers": int(top["topic_window_papers"].iloc[0]),
                "num_target_papers": int(top["topic_target_papers"].iloc[0]),
                "top_keywords": json.dumps(top["keyword"].tolist(), ensure_ascii=False),
                "top_influence_scores": json.dumps([round(float(v), 4) for v in top["influence_score"].tolist()]),
            }
        )
    topic_summary = pd.DataFrame(topic_summary_rows)
    return scores, topic_summary


def write_summary(
    outdir: Path,
    topic_summary: pd.DataFrame,
    scores: pd.DataFrame,
    window_start: int,
    window_end: int,
    target_year: int,
    top_n: int,
) -> None:
    lines = [
        f"Input years: [{window_start}, {window_start + 1 if window_start < window_end else window_start} ... {window_end}]",
        f"Target year: {target_year}",
        "",
        "Top explanatory keywords per topic:",
    ]
    for _, row in topic_summary.sort_values("topic").iterrows():
        lines.append("")
        lines.append(f"{row['topic']} | window papers={row['num_window_papers']} | target papers={row['num_target_papers']}")
        topic_scores = scores[scores["topic"] == row["topic"]].head(top_n)
        for r in topic_scores.itertuples(index=False):
            lines.append(
                f" - {r.keyword}: influence={r.influence_score:.3f}, strength={r.raw_strength:.3f}, "
                f"specificity={r.topic_specificity:.3f}, continuity={r.continuity:.3f}, trend={r.trend_ratio:.3f}"
            )
    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.window_end + 1 != args.target_year:
        print(
            f"Warning: target year {args.target_year} is not exactly window_end + 1 ({args.window_end + 1}). "
            "That is allowed for explanation, but not ideal for one-step forecasting."
        )

    ngrams = tuple(sorted({int(x.strip()) for x in args.ngrams.split(",") if x.strip()}))
    if not ngrams:
        raise ValueError("At least one ngram size is required.")
    ngram_range = (min(ngrams), max(ngrams))

    df, topic_col, text_cols = load_papers(args.input)
    docs = df["_doc_text"].tolist()

    X, vocab = build_keyword_matrix(
        docs=docs,
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_sizes=ngram_range,
    )
    paper_keywords = top_keywords_for_paper(X, vocab, args.top_keywords_per_paper)

    df_out = df.copy()
    df_out["extracted_keywords"] = [json.dumps([kw for kw, _ in kws], ensure_ascii=False) for kws in paper_keywords]
    df_out["extracted_keyword_scores"] = [json.dumps([round(float(score), 6) for _, score in kws]) for kws in paper_keywords]
    df_out.to_csv(outdir / "papers_with_keywords.csv", index=False)

    scores, topic_summary = compute_keyword_topic_influence(
        df=df,
        paper_keywords=paper_keywords,
        window_start=args.window_start,
        window_end=args.window_end,
        target_year=args.target_year,
    )

    scores.to_csv(outdir / "keyword_topic_scores.csv", index=False)
    top_per_topic = scores.groupby("topic", sort=False).head(args.top_keywords_per_topic).reset_index(drop=True)
    top_per_topic.to_csv(outdir / "top_keywords_per_topic.csv", index=False)
    topic_summary.to_csv(outdir / "topic_keyword_summary.csv", index=False)

    config = {
        "input": str(args.input),
        "window_start": args.window_start,
        "window_end": args.window_end,
        "target_year": args.target_year,
        "top_keywords_per_paper": args.top_keywords_per_paper,
        "top_keywords_per_topic": args.top_keywords_per_topic,
        "min_df": args.min_df,
        "max_features": args.max_features,
        "topic_column": topic_col,
        "text_columns_used": text_cols,
        "num_papers": int(len(df)),
        "num_keyword_features": int(len(vocab)),
    }
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_summary(
        outdir=outdir,
        topic_summary=topic_summary,
        scores=scores,
        window_start=args.window_start,
        window_end=args.window_end,
        target_year=args.target_year,
        top_n=args.top_keywords_per_topic,
    )

    print("Done.")
    print(f"Outputs written to: {outdir.resolve()}")
    print(" - papers_with_keywords.csv")
    print(" - keyword_topic_scores.csv")
    print(" - top_keywords_per_topic.csv")
    print(" - topic_keyword_summary.csv")
    print(" - run_config.json")
    print(" - summary.txt")


if __name__ == "__main__":
    main()
