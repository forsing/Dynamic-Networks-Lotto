# Dynamic Networks 
# Link Prediction 
# Regressor


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from node2vec import Node2Vec

from pyvis.network import Network

import random
from itertools import combinations



# ----------------- KREIRANJE SVIH MOGUCIH GRANA 39C7 -----------------
all_nodes = list(range(1, 40))  # cvorovi 1-39
all_combinations = list(combinations(all_nodes, 7))  # sve 39C7 kombinacije
print()
print(f"Ukupno kombinacija (39C7): {len(all_combinations)}")  # 15,380,937
print()
"""
Ukupno kombinacija (39C7): 15380937
"""


# Accessing the dataset in Python
# CSV file with header

# Učitavanje CSV fajla sa headerom
csv_path = "/Users/milan/Desktop/GHQ/data/loto7h_4530_k99.csv"
df_csv = pd.read_csv(csv_path)

print()
print(df_csv.head())
print()
"""
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     5    14    15    17    28    30    34
1     2     3    13    18    19    23    37
2    13    17    18    20    21    26    39
3    17    20    23    26    35    36    38
4     3     4     8    11    29    32    37
"""

print()
print(df_csv.dtypes)
print()
"""
Num1    int64
Num2    int64
Num3    int64
Num4    int64
Num5    int64
Num6    int64
Num7    int64
dtype: object
"""


# Descriptive Statistics
print()
print("Number of unique links in CSV:", df_csv.shape[0])
print("Number of nodes:", len(set(df_csv['Num1']).union(set(df_csv['Num7']))))
print("Node interval:", df_csv['Num1'].min(), "→", df_csv['Num7'].max())
print()
"""
Number of unique links in CSV: 4530
Number of nodes: 39
Node interval: 1 → 39
"""


# ----------------- SNAPSHOT CREATION IZ CSV-a (Samo za info) -----------------
snapshots_csv = []
for idx, row in df_csv.iterrows():
    G = nx.Graph()
    G.add_nodes_from(all_nodes)  # sve moguće čvorove
    nodes = row.values.tolist()
    # Sve moguće parove među 7 brojeva
    for u, v in combinations(nodes, 2):
        G.add_edge(u, v)
    snapshots_csv.append(G)

print()
print(f"Broj snapshotova kreiranih iz CSV fajla: {len(snapshots_csv)}")
print()
"""
Broj snapshotova kreiranih iz CSV fajla: 4530
"""


# Broj čvorova
num_nodes = 39
nodes = list(range(1, num_nodes + 1))  # Čvorovi 1,2,...,39

# Generator svih linkova (kombinacija od 7 čvorova)
all_links = combinations(nodes, 7)

print()
# Primer: ispis prvih 10 linkova
for i, link in enumerate(all_links):
    if i >= 10:
        break
    print(link)

# Ako želiš da brojiš sve linkove:
total_links = sum(1 for _ in combinations(nodes, 7))
print()
print("Ukupan broj linkova:", total_links)
print()
"""
(1, 2, 3, 4, 5, 6, 7)
(1, 2, 3, 4, 5, 6, 8)
(1, 2, 3, 4, 5, 6, 9)
(1, 2, 3, 4, 5, 6, 10)
(1, 2, 3, 4, 5, 6, 11)
(1, 2, 3, 4, 5, 6, 12)
(1, 2, 3, 4, 5, 6, 13)
(1, 2, 3, 4, 5, 6, 14)
(1, 2, 3, 4, 5, 6, 15)
(1, 2, 3, 4, 5, 6, 16)

Ukupan broj linkova: 15380937
"""


# ----------------- SNAPSHOT CREATION IZ SVIH MOGUCIH KOMBINACIJA 39C7 -----------------
"""
snapshots = []
for comb in all_combinations:
    G = nx.Graph()
    G.add_nodes_from(all_nodes)  # dodaj sve čvorove 1-39
    # Dodavanje svih međusobnih veza unutar te grane (21 veza po grani)
    for u, v in combinations(comb, 2):
        G.add_edge(u, v)
    snapshots.append(G)
"""
print()
# print(f"Broj snapshotova kreiranih iz svih kombinacija: {len(snapshots)}")
print()
"""

"""


# Broj čvorova
num_nodes = 39
nodes = list(range(1, num_nodes + 1))

# Kreiramo prazan graf
G = nx.Graph()
G.add_nodes_from(nodes)

# Generisanje svih kombinacija od 7 čvorova
all_links = combinations(nodes, 7)

# Za vizualizaciju, biramo nasumičnih 50 linkova
sample_links = random.sample(list(all_links), 50)

# Dodavanje linkova u graf
for link in sample_links:
    # Povezujemo sve čvorove unutar linka međusobno
    for i in range(len(link)):
        for j in range(i + 1, len(link)):
            G.add_edge(link[i], link[j])

# Crtanje grafičke mreže
plt.figure(figsize=(12, 12))
pos = nx.circular_layout(G)  # raspored čvorova u krug
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
plt.title("Digitalna mreža: 39 čvorova, 50 nasumičnih 7-članih linkova")
plt.show()


# ----------------- AGGREGATE GRAPH -----------------

# Aggregate graph
G_total = nx.Graph()
G_total.add_edges_from(zip(df_csv['Num1'], df_csv['Num7']))
print()
print("Number of aggregate nodes", G_total.number_of_nodes())
print("Number of aggregate arcs:", G_total.number_of_edges())
print()
"""
Number of aggregate nodes 39
Number of aggregate arcs: 289
"""


# Aggregated Network Visualization
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_total, seed=39)
nx.draw_networkx_nodes(G_total, pos, node_size=30, alpha=0.7)
nx.draw_networkx_edges(G_total, pos, alpha=0.2)
plt.title("Aggregate graph of Loto 7 dataset")
plt.axis('off')
plt.show()
"""
(A plot window will open showing the aggregate graph)
"""


# Preliminary Node Analysis
degree_dict = dict(G_total.degree())
closeness_dict = nx.closeness_centrality(G_total)
betweenness_dict = nx.betweenness_centrality(G_total, normalized=True)

# Trasformation into DataFrame
df_nodes = pd.DataFrame({
    'node': list(G_total.nodes()),
    'degree': [degree_dict[n] for n in G_total.nodes()],
    'closeness': [closeness_dict[n] for n in G_total.nodes()],
    'betweenness': [betweenness_dict[n] for n in G_total.nodes()],
})
print()
print(df_nodes.head())
print()
"""
   node  degree  closeness  betweenness
0     5      17   0.644068     0.018253
1    34      19   0.655172     0.026602
2     2      23   0.716981     0.045951
3    37      20   0.678571     0.027376
4    13      11   0.567164     0.005837
"""


# Centrality metric
degree_dict = dict(G_total.degree())
closeness_dict = nx.closeness_centrality(G_total)
betweenness_dict = nx.betweenness_centrality(G_total, normalized=True)

# Construction of dataframe nodes
df_nodes = pd.DataFrame({
    'node': list(G_total.nodes()),
    'degree': [degree_dict[n] for n in G_total.nodes()],
    'closeness': [closeness_dict[n] for n in G_total.nodes()],
    'betweenness': [betweenness_dict[n] for n in G_total.nodes()],
})

print()
print(df_nodes.head())
print()
"""
   node  degree  closeness  betweenness
0     5      17   0.644068     0.018253
1    34      19   0.655172     0.026602
2     2      23   0.716981     0.045951
3    37      20   0.678571     0.027376
4    13      11   0.567164     0.005837
"""


# Complete list of the nodes (fixed for all snapshots)
all_nodes = sorted(G_total.nodes())

# List of coherent adjacency matrices
adj_matrices = []

for G in snapshots_csv:
    G_full = nx.Graph()
    G_full.add_nodes_from(all_nodes)  
    # garantuees presence of all nodes
    G_full.add_edges_from(G.edges())  
    # add only the arcs of the current snapshot

    A = nx.to_numpy_array(G_full, nodelist=all_nodes)
    adj_matrices.append(A)


from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    preferential_attachment
)

# Choose consecutive snapshots
G_train = snapshots_csv[100]
G_test = snapshots_csv[101]

# Nodes present in the snapshot
nodes = list(G_train.nodes())

# Potential copies (not yet connected)
potential_edges = [
    (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
    if not G_train.has_edge(u, v)
]

# Common Neighbors (manual)
def compute_common_neighbors(G, pairs):
    return [(u, v, len(list(nx.common_neighbors(G, u, v)))) for u, v in pairs]

cn_scores = compute_common_neighbors(G_train, potential_edges)

# Jaccard
jaccard_scores = list(jaccard_coefficient(G_train, potential_edges))

# Adamic-Adar
aa_scores = list(adamic_adar_index(G_train, potential_edges))

# Preferential Attachment (optional)
pa_scores = list(preferential_attachment(G_train, potential_edges))

def label_links(pairs, G_future):
    return [1 if G_future.has_edge(u, v) else 0 for u, v in pairs]

labels_cn = label_links([(u, v) for u, v, _ in cn_scores], G_test)
labels_jaccard = label_links([(u, v) for u, v, _ in jaccard_scores], G_test)
labels_aa = label_links([(u, v) for u, v, _ in aa_scores], G_test)


# AUC evaluation for Jaccard:

jaccard_values = [score for _, _, score in jaccard_scores]
auc_jaccard = roc_auc_score(labels_jaccard, jaccard_values)
print()
print(f"AUC (Jaccard): {auc_jaccard:.4f}")
print()
"""
AUC (Jaccard): 0.5000
"""


# Step 1: Snapshot definition
# We choose two consecutive snapshots to build the training set:

G_train = snapshots_csv[100]
G_test = snapshots_csv[101]
nodes = list(G_train.nodes())


# Step 2: Building the candidate pairs and labels

# All the pairs not connected in the current snapshot
candidate_pairs = [
    (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
    if not G_train.has_edge(u, v)
]

# Binary label: 1 if (u,v) is present in G_test, otherwise 0
def label_links(pairs, G_future):
    return [1 if G_future.has_edge(u, v) else 0 for u, v in pairs]

labels = label_links(candidate_pairs, G_test)


# Step 3: Feature Extraction for each pair

# Function that extracts all the structural features
def extract_features(G, pairs):
    cn = { (u,v): len(list(nx.common_neighbors(G, u, v))) for u,v in pairs }
    jc = { (u,v): p for u,v,p in jaccard_coefficient(G, pairs) }
    aa = { (u,v): p for u,v,p in adamic_adar_index(G, pairs) }
    pa = { (u,v): p for u,v,p in preferential_attachment(G, pairs) }

    data = []
    for u, v in pairs:
        data.append({
            'u': u,
            'v': v,
            'cn': cn.get((u,v), 0),
            'jc': jc.get((u,v), 0),
            'aa': aa.get((u,v), 0),
            'pa': pa.get((u,v), 0),
        })
    return pd.DataFrame(data)

df_features = extract_features(G_train, candidate_pairs)
df_features['label'] = labels
print()
print(df_features.head())
print()
"""
   u  v  cn   jc  aa  pa  label
0  1  2   0  0.0   0   0      0
1  1  3   0  0.0   0   0      0
2  1  4   0  0.0   0   0      0
3  1  5   0  0.0   0   0      0
4  1  6   0  0.0   0   0      0
"""


# Step 4: Model Training and Evaluation
# Features and labels
X = df_features[['cn', 'jc', 'aa', 'pa']]
y = df_features['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=39)

# 1️⃣ Prvi structural model
reg_struct_1 = RandomForestRegressor(n_estimators=200, random_state=39, n_jobs=-1)
reg_struct_1.fit(X_train, y_train)


# Step 5: Analysis of Important Features
"""
nepostojanje informacija u label-ima 
→ svi feature-i su “nebitni” za regressor

importances ostaju praktično 0 → graf prazno izgleda

y je izuzetno neuravnotežena: 
većina vrednosti je 0, vrlo mali broj 1.

RandomForestRegressor, 
kada trenira na tako neuravnoteženim labelama 
sa malim skorom, ne uči ništa značajno 
→ svi feature importances su praktično 0 
→ graf izgleda prazan.

Koristi RandomForestClassifier umesto regressor-a
Za binarne label-e (0/1), klasifikator je prikladniji
Importances će biti vidljive
"""


# Fix the nodes on a global layout (aggregate graph)
pos = nx.spring_layout(G_total, seed=39)

# Prepare the figure
fig, ax = plt.subplots(figsize=(10, 8))

def draw_snapshot(i):
    ax.clear()
    G = snapshots_csv[i]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
    ax.set_title(f"Snapshot {i}")
    ax.axis('off')

# Generate animation
ani = animation.FuncAnimation(fig, draw_snapshot, frames=20, interval=200, repeat=False)

# Save as GIF with PillowWriter
ani.save("/Users/milan/Desktop/GHQ/forsing/QP/dynamic/animated_web.gif", writer='pillow', fps=5)
plt.show()
"""
(A GIF file named "animated_web.gif" will be created 
showing the network evolution)
"""


# === MANUAL CREATION NET===
net = Network(height='800px', width='100%', notebook=False, directed=False)


# === VISUALIZE ===
net.toggle_physics(True)
net.write_html("/Users/milan/Desktop/GHQ/forsing/QP/dynamic/predicted_net.html", open_browser=True)
"""
An HTML file named "predicted_net.html" will be created 
and opened in the browser,
showing the network with predicted links highlighted in red.
"""


# Example: Node2Vec on subsequent snapshots

# ----------------- STEP 1: Real snapshot-----------------
print("Load real snapshot ...")
G_full = snapshots_csv[10].copy()
edges_all = list(G_full.edges())
nodes = list(G_full.nodes())

# ----------------- STEP 2: Split arcs (80% train, 20% test) -----------------
print("Arc Subdivision (80% train, 20% test) ...")
random.seed(39)
random.shuffle(edges_all)
split_idx = int(0.8 * len(edges_all))
edges_train = edges_all[:split_idx]
edges_hidden = edges_all[split_idx:]

G_train = nx.Graph()
G_train.add_nodes_from(nodes)
G_train.add_edges_from(edges_train)

# ----------------- STEP 3: Node2Vec on G_train -----------------
# 2️⃣ Node2Vec embedding model
print("Node2Vec training on partial graphs ...")
node2vec = Node2Vec(G_train, dimensions=32, walk_length=10, num_walks=50, workers=2, seed=39)
model = node2vec.fit(window=5, min_count=1)


# Key string in embedding
embedding = {str(node): model.wv[str(node)] for node in G_train.nodes()}



# ----------------- STEP 4: Positive and negative copies-----------------
print("Generation of pairs and labels ...")
positive = [(u, v) for u, v in edges_hidden if str(u) in embedding and str(v) in embedding]

negatives = set()
while len(negatives) < len(positive):
    u, v = random.sample(nodes, 2)
    if not G_full.has_edge(u, v) and str(u) in embedding and str(v) in embedding:
        negatives.add((u, v))

negatives = list(negatives)
pairs = positive + negatives
labels = [1] * len(positive) + [0] * len(negatives)

# ----------------- STEP 5: Feature from embedding -----------------
def edge_embedding(u, v):
    return np.concatenate([embedding[str(u)], embedding[str(v)]])

X = np.array([edge_embedding(u, v) for u, v in pairs])
y = np.array(labels)

print()
print("Total paris:", len(X))
print("Label distribution:")
print(pd.Series(y).value_counts())
print()

# ----------------- STEP 6: Train/Test and RF -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=39)

# 3️⃣ Treći structural model (poslednji, za 7-članu predikciju)
reg_struct_2 = RandomForestRegressor(n_estimators=200, random_state=39, n_jobs=-1)
reg_struct_2.fit(X_train, y_train)

"""
Load real snapshot ...
Arc Subdivision (80% train, 20% test) ...
Node2Vec training on partial graphs ...
Computing transition probabilities: 100%|█| 39/39 [00:00<
Generating walks (CPU: 1): 100%|█| 25/25 [00:00<00:00, 84
Generating walks (CPU: 2): 100%|█| 25/25 [00:00<00:00, 84
Generation of pairs and labels ...
Total paris: 10
Label distribution:
1    5
0    5
Name: count, dtype: int64
"""


# Example: co-class-based graph
G_class = nx.Graph()
same_class = df_csv[df_csv['Num1'] == df_csv['Num2']]
G_class.add_edges_from(zip(same_class['Num1'], same_class['Num2']))



# ================== 7-NODE LINK PREDICTION ==================

G_current = snapshots_csv[-1]
nodes = list(G_current.nodes())

# 1. Skor za sve nepostojeće ivice
candidate_pairs = [
    (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
    if not G_current.has_edge(u, v)
]

df_edges = extract_features(G_current, candidate_pairs)
X_edges = df_edges[['cn', 'jc', 'aa', 'pa']]

X_edges = X_edges.values

X_edges = pd.DataFrame(X_edges, columns=['cn', 'jc', 'aa', 'pa'])
df_edges['score'] = reg_struct_1.predict(X_edges)
"""
Ako je tvoja 7-grana bazirana na CN / JC / AA / PA 
(što trenutno jeste), koristi reg_struct_1 strukturni model:
"""


# Mapa skorova po ivici
edge_score = {
    (row.u, row.v): row.score
    for _, row in df_edges.iterrows()
}

# 2. Evaluacija 7-članih grana
best_combo = None
best_score = -np.inf

# radi brzine: uzimamo samo top-N ivica
TOP_EDGES = 200
top_edges = sorted(edge_score.items(), key=lambda x: x[1], reverse=True)[:TOP_EDGES]

# skup čvorova koji se često pojavljuju
candidate_nodes = sorted(
    set(u for (u, v), _ in top_edges) |
    set(v for (u, v), _ in top_edges)
)

# generiši samo relevantne 7-kombinacije
for combo in combinations(candidate_nodes, 7):
    pairs = list(combinations(combo, 2))

    scores = []
    for u, v in pairs:
        key = (u, v) if (u, v) in edge_score else (v, u)
        if key in edge_score:
            scores.append(edge_score[key])

    if len(scores) < 21:
        continue

    combo_score = np.mean(scores)

    if combo_score > best_score:
        best_score = combo_score
        best_combo = combo


# Pretvaranje u int
best_combo_int = tuple(int(x) for x in best_combo)
print()
print("PREDIKCIJA SLEDEĆE 7-ČLANE GRANE:")
print(best_combo_int)
print("Skor:", best_score)
print()
"""
PREDIKCIJA SLEDEĆE 7-ČLANE GRANE:
(1, 2, 3, 6, 8, 9, 10)
Skor: 0.02712301587301588
"""