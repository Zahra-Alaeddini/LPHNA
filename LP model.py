import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score, roc_auc_score,
                              average_precision_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TENSOR_PATH    = 'enriched_Compound_Disease.npy'
NODE_INDEX_PATH = 'node_index.npy'
h_dim          = 64
out_dim        = 32
neg_ratio      = 0.5
num_epochs     = 500
lr             = 0.005
weight_decay   = 1e-4
dropout        = 0.1
edge_drop_rate = 0.1


# ============================ Data ============================
tensor      = np.load(TENSOR_PATH)
node_labels = np.load(NODE_INDEX_PATH)
relation_types = ['CtD', 'CpD', 'Compound_Similarity', 'Disease_Similarity']
num_rels = len(relation_types)

le_nodes     = LabelEncoder()
node_indices = le_nodes.fit_transform(node_labels)
node_mapping = {label: idx for idx, label in enumerate(le_nodes.classes_)}
num_nodes    = len(node_labels)

edge_index_list = []
edge_type_list  = []
for rel_idx in range(num_rels):
    slice_matrix = tensor[:, :, rel_idx]
    src, dst = np.nonzero(slice_matrix)
    edge_index_list.extend([[s, d] for s, d in zip(src, dst)])
    edge_type_list.extend([rel_idx] * len(src))

edge_index    = torch.tensor(edge_index_list, dtype=torch.long, device=device)
edge_type     = torch.tensor(edge_type_list,  dtype=torch.long, device=device)
all_pos_links = torch.cat([edge_index, edge_type.unsqueeze(1)], dim=1)

full_positive_set = set(map(tuple, all_pos_links.cpu().numpy()))
print(f"Total nodes: {num_nodes:,} | Total edges: {len(all_pos_links):,}")


# ============================ Split ============================
pos_links_np = all_pos_links.cpu().numpy()
rel_labels   = pos_links_np[:, 2]

similarity_rel_ids = [2, 3]
target_rel_mask = ~np.isin(rel_labels, similarity_rel_ids)
target_idx = np.where(target_rel_mask)[0]
non_target_idx = np.where(~target_rel_mask)[0]

train_target_idx, temp_idx = train_test_split(
    target_idx, test_size=0.4, stratify=rel_labels[target_idx], random_state=42)
val_target_idx, test_target_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=rel_labels[temp_idx], random_state=42)

train_idx = np.concatenate([train_target_idx, non_target_idx])
np.random.shuffle(train_idx)

train_pos = all_pos_links[torch.from_numpy(train_idx).to(device)]
val_pos   = all_pos_links[torch.from_numpy(val_target_idx).to(device)]
test_pos  = all_pos_links[torch.from_numpy(test_target_idx).to(device)]

print(f"Train edges: {len(train_pos):,} | Val: {len(val_pos):,} | Test: {len(test_pos):,}")

# ============================ Models ============================
class MultiBehaviorGNNEncoder(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim=64, out_dim=32,
                 dropout=0.1, drop_rate=0.1):
        super().__init__()
        self.num_rels  = num_rels
        self.h_dim = h_dim
        self.drop_rate = drop_rate

        self.embedding = nn.Embedding(num_nodes, h_dim)
        self.rel_linears = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for _ in range(num_rels)])
        self.att_rel = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for _ in range(num_rels)])
        self.att_node = nn.Linear(h_dim, h_dim)
        self.att_vector = nn.Parameter(torch.randn(h_dim))
        self.fc_combined = nn.Linear(h_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for lin in self.rel_linears + self.att_rel:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.att_node.weight)
        nn.init.normal_(self.att_vector, mean=0.0, std=0.1)

    def forward(self, edge_index, edge_type, num_nodes):
        if self.training and self.drop_rate > 0:
            mask = torch.rand(edge_index.size(0),
                                    device=edge_index.device) < (1 - self.drop_rate)
            edge_index = edge_index[mask]
            edge_type  = edge_type[mask]

        x    = self.embedding.weight
        msgs = torch.zeros(
            (num_nodes, self.num_rels, self.h_dim), device=device)

        for rel_id in range(self.num_rels):
            rel_edges = (edge_type == rel_id).nonzero(as_tuple=True)[0]
            if rel_edges.numel() == 0:
                continue
            src = edge_index[rel_edges, 0]
            dst = edge_index[rel_edges, 1]
            msgs[:, rel_id, :].index_add_(
                0, dst, F.relu(self.rel_linears[rel_id](x[src])))

        att_r = torch.stack(
            [F.relu(self.att_rel[r](msgs[:, r]))
             for r in range(self.num_rels)], dim=1)
        att_n  = self.att_node(x).unsqueeze(1)
        logits = torch.matmul(
            F.leaky_relu(att_r + att_n),
            self.att_vector.unsqueeze(1)).squeeze(-1)
        has_msg = (msgs.abs().sum(dim=2) > 0).float()
        logits  = logits + (1.0 - has_msg) * (-1e9)
        alphas  = F.softmax(logits, dim=1).unsqueeze(2)
        agg = torch.sum(alphas * msgs, dim=1)

        x = F.relu(self.bn1(agg))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc_combined(x)))
        return x


class MLPDecoder(nn.Module):
    def __init__(self, out_dim, num_rels, hidden_dim=64):
        super().__init__()
        self.rel_embeddings = nn.Embedding(num_rels, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def forward(self, src_embeds, rel_ids, dst_embeds):
        rel_embeds = self.rel_embeddings(rel_ids)
        combined   = torch.cat([src_embeds, rel_embeds, dst_embeds], dim=1)
        return self.mlp(combined).squeeze()


# ============================ Utils ============================
def create_supervised_links(pos_links, full_positive_set,
                             num_nodes, num_rels, neg_ratio=0.5):
    num_pos = pos_links.size(0)
    num_neg = int(num_pos * neg_ratio)

    pos_src = pos_links[:, 0].cpu().numpy()
    pos_dst = pos_links[:, 1].cpu().numpy()

    negative_links, attempts = [], 0
    max_attempts = num_neg * 10

    while len(negative_links) < num_neg and attempts < max_attempts:
        attempts += 1
        neg_src = int(random.choice(pos_src))
        neg_dst = (int(random.choice(pos_dst))
                   if random.random() < 0.5
                   else random.randint(0, num_nodes - 1))
        neg_rel = random.randint(0, num_rels - 1)
        if (neg_src, neg_dst, neg_rel) not in full_positive_set:
            negative_links.append([neg_src, neg_dst, neg_rel])

    negative_links = torch.tensor(
        negative_links[:num_neg], dtype=torch.long, device=device)

    pos_labels = torch.ones(num_pos,              device=device)
    neg_labels = torch.zeros(len(negative_links), device=device)
    all_links  = torch.cat([pos_links, negative_links], dim=0)
    all_labels = torch.cat([pos_labels, neg_labels],    dim=0)

    perm  = torch.randperm(all_links.size(0), device=device)
    all_links  = all_links[perm]
    all_labels = all_labels[perm]
    return torch.cat([all_links, all_labels.unsqueeze(1)], dim=1)


def bootstrap_ci(y_true, y_pred_or_scores, metric_func,
                 n_resamples=1000, confidence=0.95, **kwargs):
    def stat_func(yt, yp):
        try:
            if metric_func in [roc_auc_score, average_precision_score]:
                return metric_func(yt, yp, **kwargs)
            return metric_func(yt, (yp > 0.5).astype(int), **kwargs)
        except Exception:
            return np.nan

    y_true = np.asarray(y_true)
    y_pred_or_scores = np.asarray(y_pred_or_scores)
    res = bootstrap(
        (y_true, y_pred_or_scores), stat_func,
        n_resamples=n_resamples, confidence_level=confidence,
        method='percentile', random_state=42)
    return stat_func(y_true, y_pred_or_scores), res.confidence_interval


def evaluate(links, node_embeddings, decoder,
             return_scores=False, n_boot=800):
    decoder.eval()
    with torch.no_grad():
        src = links[:, 0].long()
        dst = links[:, 1].long()
        rel = links[:, 2].long()
        labels = links[:, 3].float()

        logits = decoder(node_embeddings[src], rel, node_embeddings[dst])
        scores = torch.sigmoid(logits)

        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        preds_np  = (scores_np > 0.5).astype(int)

    metrics, ci_dict = {}, {}
    metric_configs = {
        'precision': (precision_score, preds_np,  {'zero_division': 0}),
        'recall':    (recall_score,    preds_np,  {'zero_division': 0}),
        'f1':        (f1_score,        preds_np,  {'zero_division': 0}),
        'accuracy':  (accuracy_score,  preds_np,  {}),
        'roc_auc':   (roc_auc_score,   scores_np, {}),
        'aupr':      (average_precision_score, scores_np, {}),
    }
    for name, (fn, inp, kw) in metric_configs.items():
        point, ci = bootstrap_ci(labels_np, inp, fn, n_resamples=n_boot, **kw)
        metrics[name]          = point
        ci_dict[f"{name}_ci"]  = ci

    if return_scores:
        return metrics, ci_dict, scores_np, labels_np
    return metrics, ci_dict


# ============================ Training ============================
train_links = create_supervised_links(
    train_pos, full_positive_set, num_nodes, num_rels, neg_ratio)
val_links   = create_supervised_links(
    val_pos,   full_positive_set, num_nodes, num_rels, neg_ratio)
test_links  = create_supervised_links(
    test_pos,  full_positive_set, num_nodes, num_rels, neg_ratio)

encoder = MultiBehaviorGNNEncoder(
    num_nodes, num_rels, h_dim, out_dim, dropout, edge_drop_rate).to(device)
decoder = MLPDecoder(out_dim, num_rels).to(device)

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=lr, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss(reduction='none')

best_val_auc   = 0.0
best_enc_state = None
best_dec_state = None

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    node_embeddings = encoder(train_pos[:, :2], train_pos[:, 2], num_nodes)

    src = train_links[:, 0].long()
    dst = train_links[:, 1].long()
    rel = train_links[:, 2].long()
    labels = train_links[:, 3].float()

    logits   = decoder(node_embeddings[src], rel, node_embeddings[dst])
    loss_per = criterion(logits, labels)
    weights = torch.ones_like(labels)
    weights[labels == 0] = 1.0 / neg_ratio if neg_ratio > 1 else 1.0
    loss = (loss_per * weights).mean()

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        encoder.eval()
        with torch.no_grad():
            val_emb = encoder(train_pos[:, :2], train_pos[:, 2], num_nodes)
            val_metrics, _ = evaluate(val_links, val_emb, decoder, n_boot=200)
        print(f"[Epoch {epoch + 1:03d}] Loss: {loss.item():.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f} | "
              f"AUPR: {val_metrics['aupr']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc   = val_metrics['roc_auc']
            best_enc_state = {k: v.clone()
                              for k, v in encoder.state_dict().items()}
            best_dec_state = {k: v.clone()
                              for k, v in decoder.state_dict().items()}

# ============================ Final Test ============================
if best_enc_state:
    encoder.load_state_dict(best_enc_state)
    decoder.load_state_dict(best_dec_state)

encoder.eval()
with torch.no_grad():
    node_embeddings = encoder(train_pos[:, :2], train_pos[:, 2], num_nodes)
    test_metrics, test_ci, scores_np, labels_np = evaluate(
        test_links, node_embeddings, decoder,
        return_scores=True, n_boot=1000)

print("\n" + "=" * 80)
print("FINAL TEST RESULTS WITH 95% CONFIDENCE INTERVALS")
print("=" * 80)
for k in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'aupr']:
    v       = test_metrics[k]
    lo, hi  = test_ci[f"{k}_ci"]
    half_ci = (hi - lo) / 2
    print(f"{k.capitalize():12}: {v:.4f} ± {half_ci:.4f}   "
          f"95% CI: [{lo:.4f}, {hi:.4f}]")

# ============================ Save Predictions ============================
inv_rel_mapping = {i: r for i, r in enumerate(relation_types)}
mask = (scores_np >= 0.5) & (labels_np == 0)

new_results = pd.DataFrame({
    'Source':   [le_nodes.inverse_transform([int(s.item())])[0]
                 for s in test_links[:, 0][mask]],
    'Target':   [le_nodes.inverse_transform([int(d.item())])[0]
                 for d in test_links[:, 1][mask]],
    'Relation': [inv_rel_mapping[int(r.item())]
                 for r in test_links[:, 2][mask]],
    'Score':    scores_np[mask],
})

new_results.to_csv('new_predicted_links.csv', index=False)
print(f"\n{len(new_results)} new predicted links saved → new_predicted_links.csv")