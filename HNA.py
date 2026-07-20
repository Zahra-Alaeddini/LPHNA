"""
Hierarchical Neuro-Attention Explainability Scoring
for Compound-Disease Link Prediction via Metapaths

"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
from transformers import BertModel, BertTokenizer


# ====================== Reproducibility ======================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== Biomedical Language Model ======================

def load_bert() -> Tuple[BertTokenizer, BertModel]:
    print("Loading BioBERT...")
    tok = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    mdl = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
    print("BioBERT loaded successfully.")
    return tok, mdl


tokenizer, bert_model = load_bert()
bert_model.eval()


# ====================== Data Loading ======================

tensor      = np.load("enriched_Compound_Disease.npy")
node_labels = np.load("node_index.npy")

RELATION_TYPES = ["CtD", "CpD", "Compound_Similarity", "Disease_Similarity"]
rel_mapping: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}

preds_df    = pd.read_csv("new_predicted_links.csv")
metapath_df = pd.read_csv("compound_disease_metapaths.tsv", sep="\t")


# ====================== Train / Val / Test Split ======================

print("Splitting data  →  Train 60% | Val 20% | Test 20%")
train_df, _tmp   = train_test_split(preds_df, test_size=0.40, random_state=42)
val_df,  test_df = train_test_split(_tmp,     test_size=0.50, random_state=42)
print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

# ====================== Graph Construction ======================

G_nx = nx.DiGraph()
for path in metapath_df["Node_Path"]:
    nodes = path.split("->")
    for i in range(len(nodes) - 1):
        G_nx.add_edge(nodes[i], nodes[i + 1])

node_map: Dict[str, int] = {n: i for i, n in enumerate(node_labels)}

G_data           = from_networkx(G_nx)
G_data.num_nodes = len(node_labels)

edge_index: List[List[int]] = []
edge_type:  List[int]       = []
for rel_idx, rel_name in enumerate(RELATION_TYPES):
    nz = np.nonzero(tensor[:, :, rel_idx])
    for src, dst in zip(nz[0], nz[1]):
        edge_index.append([src, dst])
        edge_type.append(rel_mapping[rel_name])

G_data.edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
G_data.edge_type  = torch.tensor(edge_type,  dtype=torch.long, device=device)


# ====================== Structural Feature Caches ======================

def precompute_structural_caches(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    undirected = G.to_undirected()
    comm_map: Dict[str, int] = {}
    for idx, comp in enumerate(nx.connected_components(undirected)):
        for node in comp:
            comm_map[node] = idx
    return comm_map, dict(G.degree())


comm_map, degree_cache = precompute_structural_caches(G_nx)

# ====================== Embedding Utilities ======================

node_embedding_cache: Dict[str, torch.Tensor] = {}
metapath_embedding_cache: Dict[str, torch.Tensor] = {}


def embed_node(node: str, max_length: int = 64) -> torch.Tensor:
    if node in node_embedding_cache:
        return node_embedding_cache[node]
    text   = node.split("::")[1] if "::" in node else node
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
    with torch.no_grad():
        emb = bert_model(**inputs).last_hidden_state[:, 0, :].squeeze(0)
    node_embedding_cache[node] = emb
    return emb


def embed_metapath(path: str, max_length: int = 64) -> torch.Tensor:
    if path in metapath_embedding_cache:
        return metapath_embedding_cache[path]
    texts  = [n.split("::")[1] for n in path.split("->") if "::" in n]
    inputs = tokenizer(" ".join(texts), return_tensors="pt",
                       max_length=max_length, truncation=True, padding=True).to(device)
    with torch.no_grad():
        emb = bert_model(**inputs).last_hidden_state[:, 0, :].squeeze(0)
    metapath_embedding_cache[path] = emb
    return emb


# ====================== Feature Extraction ======================

def extract_metapath_features(source: str, target: str) -> Tuple[List[str], List[str]]:
    mask    = (metapath_df["Compound"] == source) & (metapath_df["Disease"] == target)
    matched = metapath_df[mask]
    return matched["Node_Path"].tolist(), matched["Metapath"].tolist()


def struct_features(source: str, target: str, G: nx.DiGraph = None) -> torch.Tensor:
    if G is None:
        G = G_nx
    deg_s  = degree_cache.get(source, 0)
    deg_t  = degree_cache.get(target, 0)
    nb_s   = (set(G.successors(source)) | set(G.predecessors(source))) if source in G else set()
    nb_t   = (set(G.successors(target)) | set(G.predecessors(target))) if target in G else set()
    common = len(nb_s & nb_t)
    np2    = len(list(nx.all_simple_paths(G, source, target, cutoff=2))) \
             if source in G and target in G else 0
    np3    = len(list(nx.all_simple_paths(G, source, target, cutoff=3))) \
             if source in G and target in G else 0
    same_comm = int(
        comm_map.get(source) is not None and
        comm_map.get(source) == comm_map.get(target))
    return torch.tensor([deg_s, deg_t, common + np2 + np3 + same_comm], dtype=torch.float32, device=device)

# ====================== Models ======================

class PathAttributionScorer(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),)

    def forward(self, path_embs: torch.Tensor) -> torch.Tensor:
        return self.net(path_embs).squeeze(-1)


class HierarchicalNeuroAttention(nn.Module):
    def __init__(self, semantic_dim: int = 768, structural_dim: int = 3,
                 hidden_dim: int = 128, num_heads: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.local_attn  = nn.MultiheadAttention(semantic_dim, 16, batch_first=True, dropout=dropout)
        self.local_norm  = nn.LayerNorm(semantic_dim)
        self.global_attn = nn.MultiheadAttention(hidden_dim,  num_heads, batch_first=True, dropout=dropout)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn  = nn.MultiheadAttention(hidden_dim,  num_heads, batch_first=True, dropout=dropout)
        self.cross_norm  = nn.LayerNorm(hidden_dim)

        self.semantic_ffn = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.structural_ffn = nn.Sequential(
            nn.Linear(structural_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

        self.hebbian_lr = 0.01
        self.hebbian_weights = nn.Parameter(torch.ones(semantic_dim), requires_grad=False)

    def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        delta = torch.mean(pre.squeeze(1) * post.squeeze(1), dim=0)
        self.hebbian_weights.data = torch.clamp(self.hebbian_weights.data + self.hebbian_lr * delta, 0.1, 10.0)

    def forward(self,
        semantic_embs:   torch.Tensor,
        structural_embs: torch.Tensor,
        memory_embs:     torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = semantic_embs.unsqueeze(1) * self.hebbian_weights.view(1, 1, -1)
        local_out, _ = self.local_attn(x, x, x)
        local_out  = self.local_norm(local_out + x).squeeze(1)
        self.hebbian_update(x, local_out.unsqueeze(1))
        sem_h = self.semantic_ffn(local_out)
        str_h = self.structural_ffn(structural_embs)

        combined = (sem_h + self.semantic_ffn(memory_embs.squeeze(1))) / 2 \
                   if memory_embs is not None else sem_h
        g = combined.unsqueeze(0)
        global_out, _ = self.global_attn(g, g, g)
        global_out  = self.global_norm(global_out + g).squeeze(0)

        str_h_rep  = str_h.repeat(global_out.size(0), 1).unsqueeze(0)
        cross_out, _ = self.cross_attn(global_out.unsqueeze(0), str_h_rep, str_h_rep)
        cross_out  = self.cross_norm(cross_out + global_out).squeeze(0)
        cross_out  = self.dropout(cross_out)

        policy = torch.clamp(self.policy_head(cross_out).squeeze(-1), -10, 10)
        return (torch.sigmoid(policy),
                torch.softmax(policy, dim=-1),
                self.value_head(cross_out).squeeze(-1),
                global_out.mean(dim=0))


def combined_score(aw, probs, vals, g_norm):
    return 0.4 * aw + 0.3 * probs + 0.2 * vals + 0.1 * g_norm

# ====================== Explainability Metrics ======================

def compute_comprehensiveness(
    model: HierarchicalNeuroAttention,
    path_embs: torch.Tensor,
    str_feat: torch.Tensor,
    attn_np: np.ndarray,
    orig_scores: torch.Tensor,
    top_k: int = 2,) -> float:

    model.eval()
    with torch.no_grad():
        masked = path_embs.clone()
        for k in np.argsort(attn_np)[-top_k:]:
            masked[k] = 0.0
        aw, p, v, gm = model(masked, str_feat)
        masked_scores = combined_score(aw, p, v, gm.norm())
        return float(torch.mean(torch.abs(orig_scores - masked_scores)).item())

def compute_robustness(
    model: HierarchicalNeuroAttention,
    path_embs: torch.Tensor,
    str_feat: torch.Tensor,
    original_scores: np.ndarray,
    noise_std: float = 0.1,
    num_perturbations: int = 5,) -> float:

    model.train()
    total = 0.0
    for _ in range(num_perturbations):
        scale = torch.norm(path_embs, dim=-1, keepdim=True).mean().item()
        noise = torch.normal(0., noise_std * scale, size=path_embs.shape, device=device)
        perturbed = path_embs + noise
        with torch.no_grad():
            aw, p, v, gm = model(perturbed, str_feat)
            s = np.atleast_1d(combined_score(aw, p, v, gm.norm()).cpu().numpy())
        total += np.mean(np.abs(s - original_scores))
    model.eval()
    avg_change = total / num_perturbations
    baseline   = np.mean(np.abs(original_scores)) + 1e-8
    return float(1.0 - min(avg_change / baseline, 1.0))


def compute_description_accuracy(
    narrative: str,
    src: str,
    tgt: str,
    model: HierarchicalNeuroAttention,) -> float:
    inputs = tokenizer(narrative, return_tensors="pt", max_length=64, truncation=True, padding=True).to(device)
    with torch.no_grad():
        narr_emb = bert_model(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        narr_emb = model.semantic_ffn(narr_emb).squeeze()
        src_emb  = model.semantic_ffn(embed_node(src)).squeeze()
        tgt_emb  = model.semantic_ffn(embed_node(tgt)).squeeze()
    avg = (src_emb + tgt_emb) / 2
    return max(float(1 - cosine(narr_emb.cpu().numpy(), avg.cpu().numpy())), 0.0)


def compute_stability(
    model: HierarchicalNeuroAttention,
    path_embs: torch.Tensor,
    str_feat: torch.Tensor,
    num_runs: int = 10,
    noise_std: float = 0.1,) -> float:
    model.train()
    run_means: List[float] = []
    for _ in range(num_runs):
        scale = torch.norm(path_embs, dim=-1, keepdim=True).mean().item()
        noise = torch.normal(0., noise_std * scale, size=path_embs.shape, device=device)
        with torch.no_grad():
            aw, p, v, gm = model(path_embs + noise, str_feat)
            s = combined_score(aw, p, v, gm.norm()).cpu().numpy()
        run_means.append(float(np.mean(np.atleast_1d(s))))
    model.eval()

    arr  = np.array(run_means)
    mean = np.mean(np.abs(arr))
    std  = np.std(arr)
    return float(1.0 - min(std / (mean + 1e-8), 1.0))


def compute_consistency_per_instance(
    model: HierarchicalNeuroAttention,
    all_path_embs: List[torch.Tensor],
    all_str_feats: List[torch.Tensor],
    num_runs: int = 5,
    noise_std: float = 0.1,) -> Tuple[float, List[float]]:
    model.train()
    all_sims: List[float] = []

    for path_embs, str_feat in zip(all_path_embs, all_str_feats):
        runs: List[np.ndarray] = []
        scale = torch.norm(path_embs, dim=-1, keepdim=True).mean().item()
        for _ in range(num_runs):
            noise = torch.normal(0., noise_std * scale,
                                 size=path_embs.shape, device=device)
            with torch.no_grad():
                aw, p, v, _ = model(path_embs + noise, str_feat)
            vec = np.concatenate([np.atleast_1d(aw.cpu().numpy()), np.atleast_1d(p.cpu().numpy()), np.atleast_1d(v.cpu().numpy()),])
            runs.append(vec)
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                ni = np.linalg.norm(runs[i])
                nj = np.linalg.norm(runs[j])
                if ni > 1e-8 and nj > 1e-8:
                    sim = max(float(1 - cosine(runs[i], runs[j])), 0.)
                    all_sims.append(sim)

    model.eval()
    mean_val = float(np.mean(all_sims)) if all_sims else 1.0
    return mean_val, all_sims


def compute_mean_ci(
    values: List[float], n_resamples: int = 1000, confidence: float = 0.95,) -> Tuple[float, Tuple[float, float]]:
    arr = np.array(values)
    if len(arr) < 2:
        return float(np.mean(arr)), (float("nan"), float("nan"))
    res = bootstrap((arr,), np.mean, n_resamples=n_resamples, confidence_level=confidence, method="percentile", random_state=42)
    return float(np.mean(arr)), (res.confidence_interval.low, res.confidence_interval.high)

# ====================== Model Initialization ======================

path_scorer = PathAttributionScorer().to(device)
neuro_attention = HierarchicalNeuroAttention(dropout=0.05, num_heads=16).to(device)
optimizer = torch.optim.AdamW(path_scorer.parameters(), lr=1e-4, weight_decay=1e-5)

# ====================== Training PathAttributionScorer ======================

print("\n" + "=" * 70)
print("Training PathAttributionScorer  (max 100 epochs, early stopping)")
print("=" * 70)

best_val_loss = float("inf")
patience = 15
patience_counter = 0

for epoch in range(100):
    path_scorer.train()
    train_loss, train_count = 0.0, 0
    for _, row in train_df.iterrows():
        src, tgt = row["Source"], row["Target"]
        original_score = float(row["Score"])
        paths, _ = extract_metapath_features(src, tgt)

        if not paths or src not in node_map or tgt not in node_map:
            continue

        path_embs = torch.stack([embed_metapath(p) for p in paths])
        path_scores = path_scorer(path_embs)
        target_scores = torch.full_like(path_scores, original_score)
        loss = nn.functional.mse_loss(path_scores, target_scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(path_scorer.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss  += loss.item()
        train_count += 1

    avg_train = train_loss / train_count if train_count else 0.0

    if (epoch + 1) % 5 == 0 or epoch == 99:
        path_scorer.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for _, row in val_df.iterrows():
                src, tgt  = row["Source"], row["Target"]
                paths, _  = extract_metapath_features(src, tgt)
                if not paths:
                    continue
                path_embs = torch.stack([embed_metapath(p) for p in paths])
                path_scores = path_scorer(path_embs)
                target_scores = torch.full_like(path_scores, float(row["Score"]))
                val_loss += nn.functional.mse_loss(path_scores, target_scores).item()
                val_count += 1

        avg_val = val_loss / val_count if val_count else 0.0
        print(f" Epoch {epoch+1:3d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}", end="")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(path_scorer.state_dict(), "best_path_scorer.pth")
            print(" ← best saved", end="")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n Early stopping at epoch {epoch + 1}")
                break
        print()

print(f"\nTraining complete.  Best Val Loss: {best_val_loss:.6f}")

path_scorer.load_state_dict(torch.load("best_path_scorer.pth", map_location=device))
path_scorer.eval()
neuro_attention.eval()


# ====================== Test Set Evaluation ======================

print("\n" + "=" * 70)
print("Explainability Evaluation on Test Set")
print("=" * 70)

results: List[Dict] = []
metrics: Dict[str, List[float]] = {"comprehensiveness": [], "robustness": [], "description_accuracy": [], "stability": [],}

all_path_embs_list: List[torch.Tensor] = []
all_str_feats_list: List[torch.Tensor] = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        src, tgt   = row["Source"], row["Target"]
        orig_score = float(row["Score"])
        if src not in node_map or tgt not in node_map:
            continue
        paths, metas = extract_metapath_features(src, tgt)
        if not paths:
            continue

        path_embs = torch.stack([embed_metapath(p) for p in paths])
        str_feat  = struct_features(src, tgt, G_nx).unsqueeze(0)
        learned_scores = path_scorer(path_embs)
        attn_w, probs, vals, global_mean = neuro_attention(path_embs, str_feat)
        comb = combined_score(attn_w, probs, vals, global_mean.norm())
        comb_tensor = comb.to(device)
        comb_np = np.atleast_1d(comb.cpu().numpy())
        attn_np = attn_w.cpu().numpy()
        all_path_embs_list.append(path_embs)
        all_str_feats_list.append(str_feat)

        narrative = (
            f"Link {src}->{tgt} via {metas[0] if metas else 'N/A'} "
            f"(attn:{float(attn_np.mean()):.3f}, "
            f"prob:{float(probs.cpu().numpy().mean()):.3f}, "
            f"val:{float(vals.cpu().numpy().mean()):.3f})."
        )

        comp = compute_comprehensiveness(neuro_attention, path_embs, str_feat, attn_np, comb_tensor)
        rob = compute_robustness(neuro_attention, path_embs, str_feat, comb_np)
        desc_acc = compute_description_accuracy(narrative, src, tgt, neuro_attention)
        stab = compute_stability(neuro_attention, path_embs, str_feat)

        metrics["comprehensiveness"].append(comp)
        metrics["robustness"].append(rob)
        metrics["description_accuracy"].append(desc_acc)
        metrics["stability"].append(stab)

        best_idx = int(learned_scores.argmax().item())
        results.append({
            "Source": src,
            "Target": tgt,
            "Original_Score": orig_score,
            "Best_Metapath": metas[best_idx] if metas else "N/A",
            "Best_Path": paths[best_idx],
            "Learned_Attr_Score": float(learned_scores[best_idx].item()),
            "Comprehensiveness": comp,
            "Robustness": rob,
            "Description_Accuracy": desc_acc,
            "Stability": stab,
            "Narrative": narrative,
        })

print("\nComputing Consistency (per-instance multi-run)...")
consistency, consistency_sims = compute_consistency_per_instance(neuro_attention, all_path_embs_list, all_str_feats_list, num_runs=5, noise_std=0.1,)

# ====================== Final Report ======================

print("\n=== Explainability Metrics on Test Set ===")

metric_order = [
    ("comprehensiveness",    "Comprehensiveness"),
    ("robustness",           "Robustness"),
    ("description_accuracy", "Description Accuracy"),
    ("stability",            "Stability"),
]

for key, label in metric_order:
    vals = metrics[key]
    if vals:
        mean_val, (ci_lo, ci_hi) = compute_mean_ci(vals)
        half_width = (ci_hi - ci_lo) / 2
        print(f"  {label:22}:  {mean_val:.4f} ± {half_width:.4f}   (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")

if consistency_sims:
    mean_cons, (ci_lo, ci_hi) = compute_mean_ci(consistency_sims)
    half_width = (ci_hi - ci_lo) / 2
    print(f"  {'Consistency':22}:  {mean_cons:.4f} ± {half_width:.4f}   (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
else:
    print(f"  {'Consistency':22}:  {consistency:.4f}   (no pairs found)")

results_df = pd.DataFrame(results)
results_df.to_csv("explainability_results.csv", index=False)
print("\nResults saved  →  explainability_results.csv")