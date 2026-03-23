#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Dynamic Networks 
# Link Prediction 
# Regressor


# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs


"""
DynamicNetworks_Lotto_v2 — poboljšana varijanta.

- Podrazumevani CSV: /data/loto7hh_4584_k23.csv (heder Num1–Num7).
- Ispravan agregat: svako izvlačenje = K7 klika (svi parovi), ne zip(Num1, Num7).
- Dinamička težina parova: decay^(starost) kao u graspe dinamičkom grafu.
- Link prediction: RandomForestRegressor na binarnim labelima (0/1); skor = predict.
- Node2Vec na težinskom/poslednjem prozoru; ensemble skor parova (struktura + embedding).
- Kandidati za sedmorku: TOP_NODES čvorova sa najvećim max-incident skorom para (ne po broju lopte 1..39).
- Bez učitavanja svih 39C7 kombinacija u RAM.
- Isti `--seed` (podrazumevano 39) + RF `n_jobs=1` + Node2Vec `workers=1` → ista predikcija pri svakom pokretanju.
- Grafika: `--plot` čuva PNG u `--out-dir`; prozor: dodaj `--show`.
    > python3 DynamicNetworks_Lotto_v2.py --plot --show

Zahtevi: pandas, numpy, networkx, scikit-learn, node2vec, matplotlib (opciono), pyvis (opciono).
"""


from __future__ import annotations

import argparse
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Podrazumevane putanje i hiperparametri (podešavanje predikcije)
# ---------------------------------------------------------------------------
DEFAULT_CSV = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4584_k23.csv")
SEED = 39
DECAY = 0.999
TOP_NODES_FOR_COMBO = 16
N2V_DIM = 48
N2V_WALK_LENGTH = 24
N2V_NUM_WALKS = 80
N2V_WINDOW = 8
N2V_CONTEXT_DRAWS = 800
ENSEMBLE_W_STRUCT = 0.55
ENSEMBLE_W_EMB = 0.45
RF_TREES = 400
RF_MAX_DEPTH = 16


def _detect_draw_columns(df: pd.DataFrame) -> list[str]:
    nums = [f"Num{i}" for i in range(1, 8)]
    if all(c in df.columns for c in nums):
        return nums
    base = [f"Num{i}" for i in range(1, 7)]
    if all(c in df.columns for c in base):
        if "Num7" in df.columns:
            return base + ["Num7"]
    return list(df.columns[:7])


def load_draws(csv_path: Path) -> list[list[int]]:
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = _detect_draw_columns(df)
    draws: list[list[int]] = []
    for _, row in df.iterrows():
        draws.append(sorted(int(row[c]) for c in cols))
    return draws


def draw_to_clique_graph(draw: list[int]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(1, 40))
    for u, v in combinations(draw, 2):
        a, b = (u, v) if u < v else (v, u)
        g.add_edge(a, b)
    return g


def build_snapshots(draws: list[list[int]]) -> list[nx.Graph]:
    return [draw_to_clique_graph(d) for d in draws]


def weighted_pair_graph(draws: list[list[int]], decay: float) -> nx.Graph:
    """Težine parova: suma decay^(T-1-t) za svako zajedničko izvlačenje."""
    t = len(draws)
    pair_w: dict[tuple[int, int], float] = {}
    for idx, nums in enumerate(draws):
        w = float(decay) ** (t - 1 - idx)
        for u, v in combinations(nums, 2):
            a, b = (u, v) if u < v else (v, u)
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + w
    g = nx.Graph()
    g.add_nodes_from(range(1, 40))
    for (u, v), wt in pair_w.items():
        if wt > 0:
            g.add_edge(u, v, weight=wt)
    return g


def snapshot_at_index(snapshots: list[nx.Graph], idx: int) -> nx.Graph:
    """Snapshot sa fiksnim čvorovima 1..39."""
    g = nx.Graph()
    g.add_nodes_from(range(1, 40))
    if 0 <= idx < len(snapshots):
        g.add_edges_from(snapshots[idx].edges())
    return g


def extract_features(
    G: nx.Graph,
    pairs: list[tuple[int, int]],
) -> pd.DataFrame:
    from networkx.algorithms.link_prediction import (
        adamic_adar_index,
        jaccard_coefficient,
        preferential_attachment,
    )

    cn = {(u, v): len(list(nx.common_neighbors(G, u, v))) for u, v in pairs}
    jc = {(u, v): p for u, v, p in jaccard_coefficient(G, pairs)}
    aa = {(u, v): p for u, v, p in adamic_adar_index(G, pairs)}
    pa = {(u, v): p for u, v, p in preferential_attachment(G, pairs)}
    rows = []
    for u, v in pairs:
        rows.append(
            {
                "u": u,
                "v": v,
                "cn": cn.get((u, v), 0),
                "jc": jc.get((u, v), 0.0),
                "aa": aa.get((u, v), 0.0),
                "pa": pa.get((u, v), 0.0),
            }
        )
    return pd.DataFrame(rows)


def label_pairs(pairs: list[tuple[int, int]], g_future: nx.Graph) -> list[int]:
    return [1 if g_future.has_edge(u, v) else 0 for u, v in pairs]


def candidate_non_edges(G: nx.Graph) -> list[tuple[int, int]]:
    nodes = list(range(1, 40))
    out: list[tuple[int, int]] = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if not G.has_edge(u, v):
                out.append((u, v))
    return out


def train_link_regressor(
    g_train: nx.Graph,
    g_test: nx.Graph,
    random_state: int,
) -> tuple[RandomForestRegressor, float]:
    pairs = candidate_non_edges(g_train)
    if len(pairs) == 0:
        raise RuntimeError("Nema kandidatskih parova za trening.")
    y = label_pairs(pairs, g_test)
    if sum(y) == 0:
        warnings.warn("Svi labeli 0 između ova dva snapshota — AUC nema smisla; koristim poslednji par validnih indeksa.")
    df = extract_features(g_train, pairs)
    X = df[["cn", "jc", "aa", "pa"]].values
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )
    reg = RandomForestRegressor(
        n_estimators=RF_TREES,
        max_depth=RF_MAX_DEPTH,
        random_state=random_state,
        n_jobs=1,  # -1 menja redosled stabala → drugačija predikcija pri svakom startu
    )
    reg.fit(X_tr, y_tr)
    auc = 0.0
    if len(set(y_te)) > 1:
        y_score = reg.predict(X_te)
        auc = float(roc_auc_score(y_te, y_score))
    reg.fit(X, y)
    return reg, auc


def graph_for_node2vec(draws: list[list[int]], n_last: int) -> nx.Graph:
    """Neponderisan graf poslednjih n_last kola (ili ceo niz) za Node2Vec."""
    h = nx.Graph()
    h.add_nodes_from(range(1, 40))
    window = draws[-n_last:] if n_last and len(draws) > n_last else draws
    for nums in window:
        for u, v in combinations(nums, 2):
            a, b = (u, v) if u < v else (v, u)
            h.add_edge(a, b)
    return h


def run_node2vec(g: nx.Graph, seed: int) -> dict[int, np.ndarray]:
    if g.number_of_edges() == 0:
        return {n: np.zeros(N2V_DIM, dtype=np.float64) for n in range(1, 40)}
    n2v = Node2Vec(
        g,
        dimensions=N2V_DIM,
        walk_length=N2V_WALK_LENGTH,
        num_walks=N2V_NUM_WALKS,
        workers=1,  # više niti → nedeterministički embedding i druga sedmorka
        seed=seed,
    )
    model = n2v.fit(window=N2V_WINDOW, min_count=1)
    emb: dict[int, np.ndarray] = {}
    for n in range(1, 40):
        s = str(n)
        emb[n] = (
            np.asarray(model.wv[s], dtype=np.float64)
            if s in model.wv
            else np.zeros(N2V_DIM, dtype=np.float64)
        )
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def predict_seven(
    g_current: nx.Graph,
    reg: RandomForestRegressor,
    emb: dict[int, np.ndarray],
    w_struct: float,
    w_emb: float,
    top_nodes: int,
) -> tuple[tuple[int, ...], float]:
    pairs = candidate_non_edges(g_current)
    df = extract_features(g_current, pairs)
    X = df[["cn", "jc", "aa", "pa"]].values
    p_struct = reg.predict(X)

    emb_s = []
    for _, row in df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        emb_s.append((cosine_sim(emb[u], emb[v]) + 1.0) / 2.0)
    emb_s = np.asarray(emb_s, dtype=np.float64)
    score = w_struct * p_struct + w_emb * emb_s
    df = df.copy()
    df["score"] = score

    edge_score: dict[tuple[int, int], float] = {}
    for _, row in df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        a, b = (u, v) if u < v else (v, u)
        edge_score[(a, b)] = float(row["score"])

    # Po svakom čvoru 1..39: maksimum skora po svim kandidatskim ne-ivicama; zatim top_nodes
    # najboljih čvorova (ne „16 najmanjih brojeva“ — sedmorka može da ide do 39).
    node_best: dict[int, float] = {n: float("-inf") for n in range(1, 40)}
    for (a, b), s in edge_score.items():
        if s > node_best[a]:
            node_best[a] = s
        if s > node_best[b]:
            node_best[b] = s
    for n in range(1, 40):
        if not np.isfinite(node_best[n]):
            node_best[n] = 0.0
    cand_list = sorted(range(1, 40), key=lambda n: (-node_best[n], n))[:top_nodes]

    best_combo: tuple[int, ...] | None = None
    best_mean = -np.inf
    for combo in combinations(cand_list, 7):
        pts = list(combinations(combo, 2))
        vals = []
        for u, v in pts:
            a, b = (u, v) if u < v else (v, u)
            vals.append(edge_score.get((a, b), 0.0))
        if len(vals) < 21:
            continue
        m = float(np.mean(vals))
        if m > best_mean or (np.isclose(m, best_mean) and best_combo is not None and combo < best_combo):
            best_mean = m
            best_combo = tuple(sorted(combo))

    if best_combo is None:
        best_combo = tuple(range(1, 8))
        best_mean = 0.0
    return best_combo, best_mean


def maybe_plots(
    g_total: nx.Graph,
    snapshots: list[nx.Graph],
    plot: bool,
    out_dir: Path,
    show_window: bool,
) -> None:
    if not plot:
        return
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(g_total, seed=SEED)
    nx.draw_networkx_nodes(g_total, pos, node_size=40, alpha=0.75)
    nx.draw_networkx_edges(g_total, pos, alpha=0.25, width=0.5)
    plt.title("DynamicNetworks_Lotto_v2 — ponderisani agregat (decay)")
    plt.axis("off")
    plt.tight_layout()
    path = out_dir / "aggregate_weighted.png"
    plt.savefig(path, dpi=150)
    print(f"[plot] Sačuvano: {path}")
    if show_window:
        plt.show()
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="DynamicNetworks_Lotto_v2 — predikcija sedmorke")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--decay", type=float, default=DECAY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Sačuvaj PNG agregata u --out-dir (podrazumevano bez prozora)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Uz --plot: otvori prozor (plt.show) nakon čuvanja PNG",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "DynamicNetworks_Lotto_v2_out",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.csv.is_file():
        print(f"Greška: ne postoji CSV: {args.csv}", file=sys.stderr)
        sys.exit(1)

    draws = load_draws(args.csv)
    t = len(draws)
    print(f"Učitano izvlačenja: {t} | CSV: {args.csv.resolve()}")
    print(f"decay={args.decay} | seed={args.seed}")

    snapshots = build_snapshots(draws)
    g_weighted = weighted_pair_graph(draws, args.decay)

    maybe_plots(g_weighted, snapshots, args.plot, args.out_dir, args.show)

    if t < 3:
        print("Premalo redova za snapshot trening — izlaz.")
        sys.exit(1)

    i_train, i_test = t - 2, t - 1
    g_tr = snapshot_at_index(snapshots, i_train)
    g_te = snapshot_at_index(snapshots, i_test)
    reg, auc = train_link_regressor(g_tr, g_te, args.seed)
    print(f"Link model: RandomForestRegressor | AUC (holdout, približno): {auc:.4f}")

    g_n2v = graph_for_node2vec(draws, N2V_CONTEXT_DRAWS)
    print(f"Node2Vec: dim={N2V_DIM}, walks={N2V_NUM_WALKS}, len={N2V_WALK_LENGTH} | ivice u grafu: {g_n2v.number_of_edges()}")
    emb = run_node2vec(g_n2v, args.seed)

    g_current = snapshot_at_index(snapshots, i_test)
    combo, sc = predict_seven(
        g_current,
        reg,
        emb,
        ENSEMBLE_W_STRUCT,
        ENSEMBLE_W_EMB,
        TOP_NODES_FOR_COMBO,
    )

    print()
    print("--- PREDIKCIJA SLEDEĆE KOMBINACIJE (sedmorka) ---")
    print(list(combo))
    print(f"Srednji ensemble skor (parovi u kombinaciji): {sc:.6f}")
    print()


if __name__ == "__main__":
    main()


"""
Učitano izvlačenja: 4584 | CSV: /data/loto7hh_4584_k23.csv
decay=0.999 | seed=39
Link model: RandomForestRegressor | AUC (holdout, približno): 0.5000
Node2Vec: dim=48, walks=80, len=24 | ivice u grafu: 741
Computing transition probabilities: 100%|█| 39/39 [00:00<00:00
Generating walks (CPU: 1): 100%|█| 80/80 [00:00<00:00, 366.99i

--- PREDIKCIJA SLEDEĆE KOMBINACIJE (sedmorka) ---
[4, 11, 17, 21, 30, 32, 37]
Srednji ensemble skor (parovi u kombinaciji): 0.462697
"""


"""
--plot samo čuva PNG 
u DynamicNetworks_Lotto_v2_out/aggregate_weighted.png 
i zatvara figure — nema plt.show(), zato se prozor ne otvara. 
Dodati opciju --show da otvori prozor nakon čuvanja.
"""
