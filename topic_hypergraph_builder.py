#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd


# ------------------------------
# Editable taxonomy + rules
# ------------------------------
# These are intentionally broad enough to cover most CS literature, while still
# allowing specific papers to receive multiple sensible topic labels.
TOPIC_RULES: Dict[str, List[str]] = {
    "Large Language Models": [
        r"\blarge language model(s)?\b",
        r"\bllm(s)?\b",
        r"\bfoundation model(s)?\b",
        r"\bgenerative pre[- ]?trained transformer\b",
        r"\binstruction tuning\b",
        r"\binstruction[- ]tuned\b",
        r"\bprompt(ing)?\b",
        r"\brlhf\b",
        r"\blanguage model(s)?\b",
        r"\bgpt[- ]?\d*\b",
        r"\bbert\b",
        r"\bllama\b",
        r"\btransformer[- ]based language model(s)?\b",
    ],
    "Natural Language Processing": [
        r"\bnatural language processing\b",
        r"\bnlp\b",
        r"\bmachine translation\b",
        r"\bsentiment (analysis|classification)\b",
        r"\btext classification\b",
        r"\btext mining\b",
        r"\binformation extraction\b",
        r"\bnamed entity recognition\b",
        r"\bquestion answering\b",
        r"\bsummarization\b",
        r"\bdialog(ue)? system(s)?\b",
        r"\bchatbot(s)?\b",
        r"\btokenization\b",
        r"\bword embedding(s)?\b",
        r"\bsemantic parsing\b",
        r"\bpart[- ]?of[- ]?speech\b",
        r"\bpos tagging\b",
        r"\bdependency parsing\b",
        r"\bparsing\b",
        r"\bsyntactic\b",
        r"\bsemantic(s)?\b",
        r"\btriple extraction\b",
        r"\brelation extraction\b",
        r"\bidentity resolution\b",
        r"\bcoreference resolution\b",
        r"\bword sense disambiguation\b",
        r"\bparagraph representation\b",
        r"\bsentence (embedding|representation)\b",
        r"\bmt evaluation\b",
        r"\brouge\b",
        r"\bbleu\b",
        r"\biomtk\b",
        r"\bautomated metric(s)?\b",
        r"\blinguistic(s)?\b",
        r"\bcorpus\b",
        r"\btf[- ]?idf\b",
        r"\bvector space model\b",
        r"\bterm frequency\b",
        r"\bidf\b",
        r"\bclinical nlp\b",
        r"\bbiomedical nlp\b",
        r"\bbio[- ]?entity recognition\b",
        r"\bnamed entity disambiguation\b",
    ],
    "Reinforcement Learning": [
        r"\breinforcement learning\b",
        r"(?<![a-z])rl(?![a-z])",
        r"\bq[- ]?learning\b",
        r"\bpolicy gradient(s)?\b",
        r"\bactor[- ]critic\b",
        r"\bmarkov decision process(es)?\b",
        r"\bbandit(s)?\b",
        r"\bexploration[- ]exploitation\b",
        r"\bvalue iteration\b",
        r"\bdeep q network(s)?\b",
        r"\breinforcement learning from human feedback\b",
        r"\brlhf\b",
    ],
    "Machine Learning": [
        r"\bmachine learning\b",
        r"\bstatistical learning\b",
        r"\bclassification\b",
        r"\bregression\b",
        r"\bsupervised learning\b",
        r"\bunsupervised learning\b",
        r"\bsemi[- ]?supervised learning\b",
        r"\btransfer learning\b",
        r"\bmetric learning\b",
        r"\bfeature selection\b",
        r"\bfeature engineering\b",
        r"\bgradient[- ]?based learning\b",
        r"\bgradient based learning\b",
        r"\blearning theory\b",
        r"\bsmote\b",
        r"\bsampling\b",
        r"\bdata augmentation\b",
        r"\bimbalanced (data|class)\b",
        r"\bweight tuning\b",
        r"\bclass (imbalance|weighting)\b",
    ],
    "Deep Learning": [
        r"\bdeep learning\b",
        r"\bdeep neural network(s)?\b",
        r"\bneural network(s)?\b",
        r"\bcnn(s)?\b",
        r"\bconvolutional neural network(s)?\b",
        r"\brnn(s)?\b",
        r"\brecurrent neural network(s)?\b",
        r"\blstm(s)?\b",
        r"\btransformer(s)?\b",
        r"\battention mechanism(s)?\b",
        r"\bbackpropagation\b",
        r"\bgradient descent\b",
        r"\blayer[- ]?wise (training|learning)\b",
        r"\bgreedy learning\b",
        r"\bpre[- ]?training\b",
        r"\bfine[- ]?tuning\b",
        r"\bactivation function(s)?\b",
        r"\brelu\b",
        r"\bsoftmax\b",
        r"\bsigmoid\b",
        r"\bvanishing gradient\b",
        r"\bweight initialization\b",
        r"\brepresentation learning\b",
    ],
    "Computer Vision": [
        r"\bcomputer vision\b",
        r"\bimage classification\b",
        r"\bimage detection\b",
        r"\bobject detection\b",
        r"\bimage segmentation\b",
        r"\bsemantic segmentation\b",
        r"\binstance segmentation\b",
        r"\bimage recognition\b",
        r"\bvisual recognition\b",
        r"\bimage retrieval\b",
        r"\bvision transformer(s)?\b",
        r"\bmedical imaging\b",
        r"\bvideo understanding\b",
        r"\bvideo classification\b",
        r"\bimage quality\b",
        r"\bpascal visual\b",
        r"\bvisual object class\b",
        r"\bvoc challenge\b",
        r"\bstructural similarity\b",
        r"\bssim\b",
        r"\bperceptual (loss|quality)\b",
    ],
    "Graph Machine Learning": [
        r"\bgraph neural network(s)?\b",
        r"\bgnn(s)?\b",
        r"\bgraph learning\b",
        r"\bgraph representation learning\b",
        r"\bgraph embedding(s)?\b",
        r"\bnode classification\b",
        r"\blink prediction\b",
        r"\bknowledge graph(s)?\b",
        r"\bhypergraph(s)?\b",
    ],
    "Generative Models": [
        r"\bgenerative model(s)?\b",
        r"\bautoencoder(s)?\b",
        r"\bvariational autoencoder(s)?\b",
        r"\bvae(s)?\b",
        r"\bgenerative adversarial network(s)?\b",
        r"\bgan(s)?\b",
        r"\bdiffusion model(s)?\b",
        r"\bdiffusion probabilistic model(s)?\b",
        r"\bflow[- ]based model(s)?\b",
    ],
    "Optimization": [
        r"\boptimization\b",
        r"\boptimisation\b",
        r"\boptimizer\b",
        r"\bstochastic gradient descent\b",
        r"\badam\b",
        r"\bconvex optimization\b",
        r"\bnonconvex optimization\b",
        r"\blagrangian\b",
        r"\bobjective function\b",
        r"\bautomatic differentiation\b",
        r"\bautodiff\b",
        r"\bgradient computation\b",
        r"\bcomputing gradient(s)?\b",
        r"\bderivative(s)?\b",
        r"\blearning rate\b",
        r"\bmomentum\b",
        r"\badagrad\b",
        r"\brmsprop\b",
    ],
    "Statistical Learning": [
        r"\bstatistical learning\b",
        r"\bstatistical learning theory\b",
        r"\bbayes(ian)?\b",
        r"\bprobabilistic model(s)?\b",
        r"\blikelihood\b",
        r"\bposterior\b",
        r"\bmaximum likelihood\b",
        r"\bmaximum a posteriori\b",
    ],
    "Kernel Methods / SVM": [
        r"\bsupport vector machine(s)?\b",
        r"\bsvm(s)?\b",
        r"\bkernel[- ]?based\b",
        r"\bkernel method(s)?\b",
        r"\bradial basis function\b",
        r"\brbf kernel\b",
    ],
    "Data Mining": [
        r"\bdata mining\b",
        r"\bknowledge discovery\b",
        r"\bclustering\b",
        r"\banomaly detection\b",
        r"\boutlier detection\b",
        r"\bassociation rule(s)?\b",
        r"\bpattern mining\b",
        r"\bfrequent itemset(s)?\b",
    ],
    "Information Retrieval": [
        r"\binformation retrieval\b",
        r"\bsearch engine(s)?\b",
        r"\bretrieval\b",
        r"\bdocument retrieval\b",
        r"\branking\b",
        r"\blearning to rank\b",
        r"\bpassage retrieval\b",
        r"\bsearch context\b",
        r"\bvector space\b",
        r"\bquery processing\b",
        r"\binformation needs\b",
        r"\bsearch relevance\b",
        r"\binformational (search|retrieval)\b",
    ],
    "Recommender Systems": [
        r"\brecommender system(s)?\b",
        r"\brecommendation system(s)?\b",
        r"\bcollaborative filtering\b",
        r"\bmatrix factorization\b",
        r"\btop[- ]?n recommendation\b",
    ],
    "Robotics and Control": [
        r"\brobotics\b",
        r"\brobot\b",
        r"\bcontrol system(s)?\b",
        r"\bcontrol policy\b",
        r"\bautonomous driving\b",
        r"\bpath planning\b",
        r"\bmotion planning\b",
        r"\bmanipulation\b",
    ],
    "Databases and Data Management": [
        r"\bdatabase(s)?\b",
        r"\bdb2\b",
        r"\bdbms\b",
        r"\bsql\b",
        r"\bquery optimization\b",
        r"\bquery processing\b",
        r"\blearning optimizer\b",
        r"\btransaction processing\b",
        r"\bdata management\b",
        r"\bvery large data base(s)?\b",
        r"\bvery large data bases conference\b",
        r"\bvldb\b",
    ],
    "Distributed Systems and Cloud": [
        r"\bdistributed system(s)?\b",
        r"\bcloud computing\b",
        r"\bcluster computing\b",
        r"\bmapreduce\b",
        r"\bspark\b",
        r"\bkubernetes\b",
        r"\bfault tolerance\b",
        r"\bconsensus\b",
        r"\bmicroservice(s)?\b",
    ],
    "Computer Networks": [
        r"\bcomputer network(s)?\b",
        r"\bnetwork protocol(s)?\b",
        r"\bcongestion control\b",
        r"\brouting\b",
        r"\btcp\b",
        r"\bwireless network(s)?\b",
        r"\binternet\b",
        r"\bthroughput\b",
        r"\blatency\b",
    ],
    "Security and Privacy": [
        r"\bsecurity\b",
        r"\bprivacy\b",
        r"\bmalware\b",
        r"\bintrusion detection\b",
        r"\bencryption\b",
        r"\bcryptography\b",
        r"\bauthentication\b",
        r"\baccess control\b",
        r"\badversarial attack(s)?\b",
        r"\bsecure\b",
    ],
    "Software Engineering": [
        r"\bsoftware engineering\b",
        r"\bprogram analysis\b",
        r"\bstatic analysis\b",
        r"\bdynamic analysis\b",
        r"\bbug prediction\b",
        r"\bsoftware defect prediction\b",
        r"\btest generation\b",
        r"\bunit testing\b",
        r"\bcode summarization\b",
        r"\bcode generation\b",
        r"\brefactoring\b",
        r"\bmining software repositories\b",
        r"\bcorrectness\b",
        r"\bsoftware testing\b",
        r"\bsystematic review\b",
        r"\bsystematic.(literature )?map\b",
        r"\bverification\b",
        r"\bformal methods\b",
        r"\bprogram verification\b",
    ],
    "Programming Languages": [
        r"\bprogramming language(s)?\b",
        r"\bcompiler(s)?\b",
        r"\btype system(s)?\b",
        r"\bstatic typing\b",
        r"\bformal semantics\b",
        r"\bprogram synthesis\b",
        r"\bcode optimization\b",
    ],
    "Theory and Algorithms": [
        r"\balgorithm(s)?\b",
        r"\bcomputational complexity\b",
        r"\bapproximation algorithm(s)?\b",
        r"\bgraph algorithm(s)?\b",
        r"\bcombinatorial optimization\b",
        r"\bformal verification\b",
        r"\bproof\b",
        r"\btheorem\b",
    ],
    "Human-Computer Interaction": [
        r"\bhuman[- ]computer interaction\b",
        r"\bhci\b",
        r"\buser interface\b",
        r"\busability\b",
        r"\buser study\b",
        r"\binteraction design\b",
    ],
    "Explainable / Fair / Causal AI": [
        r"\bexplainable ai\b",
        r"\bxai\b",
        r"\binterpretability\b",
        r"\bfairness\b",
        r"\bbias mitigation\b",
        r"\bcausal inference\b",
        r"\bcounterfactual\b",
        r"\bshapley\b",
    ],
    "Bioinformatics and Computational Biology": [
        r"\bbioinformatics\b",
        r"\bcomputational biology\b",
        r"\bgenomics\b",
        r"\bprotein structure\b",
        r"\bdrug discovery\b",
        r"\bdna\b",
        r"\brna\b",
        r"\bunified medical language system\b",
        r"\bumls\b",
        r"\bmedical terminology\b",
        r"\bbiomedical\b",
        r"\bclinical data\b",
        r"\bhealth(care)? informatics\b",
        r"\bdrug[- ]?target\b",
        r"\bmolecular\b",
        r"\bsequence analysis\b",
        r"\bgene expression\b",
        r"\bprotein function\b",
    ],
}


# Parent propagation: if a paper matches a specific topic, also attach the broader topic(s).
PARENT_TOPICS: Dict[str, List[str]] = {
    "Large Language Models": ["Natural Language Processing", "Deep Learning", "Machine Learning"],
    "Natural Language Processing": ["Machine Learning"],
    "Reinforcement Learning": ["Machine Learning"],
    "Deep Learning": ["Machine Learning"],
    "Computer Vision": ["Machine Learning"],
    "Graph Machine Learning": ["Machine Learning"],
    "Generative Models": ["Deep Learning", "Machine Learning"],
    "Optimization": ["Machine Learning"],
    "Statistical Learning": ["Machine Learning"],
    "Kernel Methods / SVM": ["Machine Learning"],
    "Data Mining": ["Machine Learning"],
    "Information Retrieval": ["Machine Learning"],
    "Recommender Systems": ["Machine Learning"],
    "Robotics and Control": ["Machine Learning"],
    "Explainable / Fair / Causal AI": ["Machine Learning"],
}


DEFAULT_FALLBACK_TOPIC = None  # Will use smart fallback instead


def smart_fallback(search_text: str) -> str:
    """Attempt intelligent categorization for unmatched papers."""
    text_lower = search_text.lower()

    # Common indicators for topic families
    if any(term in text_lower for term in ['sentiment', 'text', 'language', 'word', 'parsing',
                                             'token', 'embed', 'linguistic', 'annotation', 'corpus',
                                             'nltk', 'rouge', 'bleu', 'metric', 'summary', 'abstract',
                                             'semantic', 'syntactic', 'dialog']):
        return "Natural Language Processing"

    if any(term in text_lower for term in ['image', 'visual', 'video', 'scene', 'object', 'photo',
                                             'pixel', 'edge', 'color', 'picture', 'shape', 'feature detection']):
        return "Computer Vision"

    if any(term in text_lower for term in ['test', 'verification', 'correctness', 'validation', 'proof',
                                             'formal', 'quality', 'review', 'procedure', 'methodology']):
        return "Software Engineering"

    if any(term in text_lower for term in ['medical', 'health', 'clinical', 'hospital', 'patient',
                                             'disease', 'diagnosis', 'treatment', 'drug', 'biomedical',
                                             'genomic', 'molecular', 'protein', 'cell']):
        return "Bioinformatics and Computational Biology"

    if any(term in text_lower for term in ['search', 'query', 'retrieval', 'ranking', 'document',
                                             'index', 'database', 'database', 'sql', 'table']):
        return "Information Retrieval"

    if any(term in text_lower for term in ['network', 'distributed', 'parallel', 'concurrent', 'protocol',
                                             'communication', 'routing', 'internet', 'wireless']):
        return "Computer Networks"

    if any(term in text_lower for term in ['security', 'privacy', 'encryption', 'authentication',
                                             'attack', 'malware', 'intrusion', 'threat']):
        return "Security and Privacy"

    if any(term in text_lower for term in ['algorithm', 'complexity', 'approximation', 'graph',
                                             'combinatorial', 'computational']):
        return "Theory and Algorithms"

    # Default to Machine Learning if nothing else matches
    return "Machine Learning"


def compile_topic_rules(topic_rules: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = {}
    for topic, patterns in topic_rules.items():
        compiled[topic] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    return compiled


COMPILED_RULES = compile_topic_rules(TOPIC_RULES)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paper_to_search_text(row: pd.Series) -> str:
    parts = [
        safe_text(row.get("title", "")),
        safe_text(row.get("fields_of_study", "")),
        safe_text(row.get("venue", "")),
    ]
    return normalize_text(" | ".join(p for p in parts if p))


def propagate_parents(topics: Iterable[str]) -> Set[str]:
    expanded = set(topics)
    queue = list(topics)

    while queue:
        current = queue.pop()
        for parent in PARENT_TOPICS.get(current, []):
            if parent not in expanded:
                expanded.add(parent)
                queue.append(parent)
    return expanded


def assign_topics(search_text: str) -> tuple[List[str], Dict[str, int]]:
    matched_topics: Set[str] = set()
    match_counts: Dict[str, int] = {}

    for topic, patterns in COMPILED_RULES.items():
        hits = sum(1 for pattern in patterns if pattern.search(search_text))
        if hits > 0:
            matched_topics.add(topic)
            match_counts[topic] = hits

    matched_topics = propagate_parents(matched_topics)

    if not matched_topics:
        fallback = smart_fallback(search_text)
        matched_topics = {fallback}
        match_counts[fallback] = 1

    # Sorted for stable output.
    ordered_topics = sorted(
        matched_topics,
        key=lambda t: (-match_counts.get(t, 0), t)
    )
    return ordered_topics, match_counts


def choose_primary_topic(topics: List[str], match_counts: Dict[str, int]) -> str:
    if not topics:
        return smart_fallback("")  # Empty fallback if no topics
    return sorted(topics, key=lambda t: (-match_counts.get(t, 0), t))[0]


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Expected at least: {required}"
        )


def build_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    topic_records = []
    incidence_records = []
    hyperedges = []
    topic_counter = Counter()

    assigned_topics_col: List[List[str]] = []
    primary_topic_col: List[str] = []
    n_topics_col: List[int] = []
    search_text_col: List[str] = []

    for _, row in df.iterrows():
        search_text = paper_to_search_text(row)
        topics, match_counts = assign_topics(search_text)
        primary_topic = choose_primary_topic(topics, match_counts)

        paper_id = safe_text(row.get("paper_id", ""))
        title = safe_text(row.get("title", ""))
        year = safe_text(row.get("year", ""))

        assigned_topics_col.append(topics)
        primary_topic_col.append(primary_topic)
        n_topics_col.append(len(topics))
        search_text_col.append(search_text)

        topic_ids = []
        for topic in topics:
            topic_id = slugify(topic)
            topic_ids.append(topic_id)
            topic_counter[topic] += 1
            incidence_records.append(
                {
                    "paper_id": paper_id,
                    "topic_id": topic_id,
                    "topic_name": topic,
                    "title": title,
                    "year": year,
                }
            )

        hyperedges.append(
            {
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "citation_count": safe_text(row.get("citation_count", "")),
                "venue": safe_text(row.get("venue", "")),
                "fields_of_study": safe_text(row.get("fields_of_study", "")),
                "topics": topics,
                "topic_ids": topic_ids,
                "primary_topic": primary_topic,
            }
        )

    df = df.copy()
    df["assigned_topics"] = [json.dumps(x, ensure_ascii=False) for x in assigned_topics_col]
    df["primary_topic"] = primary_topic_col
    df["num_topics"] = n_topics_col
    df["search_text_used_for_matching"] = search_text_col

    all_topics = sorted({*TOPIC_RULES.keys()})  # Removed DEFAULT_FALLBACK_TOPIC
    for topic in all_topics:
        topic_records.append(
            {
                "topic_id": slugify(topic),
                "topic_name": topic,
                "paper_count": topic_counter.get(topic, 0),
            }
        )

    topic_nodes_df = pd.DataFrame(topic_records).sort_values(
        by=["paper_count", "topic_name"], ascending=[False, True]
    )
    incidence_df = pd.DataFrame(incidence_records)

    # Useful CSVs
    df.to_csv(outdir / "papers_with_topics.csv", index=False)
    topic_nodes_df.to_csv(outdir / "topic_nodes.csv", index=False)
    incidence_df.to_csv(outdir / "topic_paper_incidence.csv", index=False)

    # Hypergraph JSON: topic nodes + paper hyperedges.
    hypergraph = {
        "nodes": topic_records,
        "hyperedges": hyperedges,
        "metadata": {
            "num_papers": int(len(df)),
            "num_topics": int(len(topic_records)),
            "num_incidence_pairs": int(len(incidence_records)),
            "description": "Topics are nodes, papers are hyperedges, and each hyperedge connects to every assigned topic.",
        },
    }

    with open(outdir / "hypergraph.json", "w", encoding="utf-8") as f:
        json.dump(hypergraph, f, indent=2, ensure_ascii=False)

    # Optional edge-centric CSV if you want a single file per paper edge.
    pd.DataFrame(hyperedges).to_csv(outdir / "paper_hyperedges.csv", index=False)

    # Simple summary text file.
    summary_lines = [
        f"Papers processed: {len(df)}",
        f"Topic nodes: {len(topic_records)}",
        f"Paper-topic incidence pairs: {len(incidence_records)}",
        "",
        "Top topics by paper count:",
    ]
    for _, row in topic_nodes_df.head(15).iterrows():
        summary_lines.append(f"- {row['topic_name']}: {row['paper_count']}")

    (outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign multi-label CS topics to papers and export hypergraph-ready files."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--outdir",
        default="topic_hypergraph_output",
        help="Directory where output files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_columns(df, ["paper_id", "title"])

    build_outputs(df, outdir)
    print(f"Done. Wrote outputs to: {outdir.resolve()}")
    print("Files:")
    print(" - papers_with_topics.csv")
    print(" - topic_nodes.csv")
    print(" - topic_paper_incidence.csv")
    print(" - paper_hyperedges.csv")
    print(" - hypergraph.json")
    print(" - summary.txt")


if __name__ == "__main__":
    main()
