import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import os
from scipy import sparse
from joblib import Parallel, delayed

DATA_PATH = 'edges/'
NODE_LIST_PATH = 'edges/nodes.tsv'


def load_node_list():
    try:
        df = pd.read_csv(NODE_LIST_PATH, sep='\t')
        if 'id' not in df.columns or 'kind' not in df.columns:
            raise ValueError(f"Node list TSV must have 'id' and 'kind' columns. Found: {df.columns.tolist()}")
        node_types = ['Compound', 'Disease', 'Gene', 'Anatomy', 'Symptom', 'Side Effect']
        nodes = {kind: set(df[df['kind'] == kind]['id'].dropna()) for kind in node_types}
        return nodes
    except Exception as e:
        raise ValueError(f"Failed to load node list from {NODE_LIST_PATH}: {str(e)}")


def load_edges():
    edges = {}
    for file in os.listdir(DATA_PATH):
        if file.endswith('.tsv') and file != 'nodes.tsv':
            metaedge = file.replace('.tsv', '')
            df = pd.read_csv(os.path.join(DATA_PATH, file), sep='\t')
            if list(df.columns) != ['source', 'metaedge', 'target']:
                raise ValueError(f"Unexpected columns in {file}: {df.columns.tolist()}")
            if metaedge == 'CtD':
                edges['CtD'] = df[df['metaedge'] == 'CtD'][['source', 'target']]
                edges['CpD'] = df[df['metaedge'] == 'CpD'][['source', 'target']]
            else:
                edges[metaedge] = df[['source', 'target']]
    return edges


def build_graph(edges):
    G = nx.MultiDiGraph()
    for metaedge, df in edges.items():
        for _, row in df.iterrows():
            G.add_edge(row['source'], row['target'], metaedge=metaedge)
    return G


def create_adjacency_matrix(df, source_nodes, target_nodes, source_type, target_type):
    source_nodes = sorted(list(source_nodes))
    target_nodes = sorted(list(target_nodes))
    source_index = {n: i for i, n in enumerate(source_nodes)}
    target_index = {n: i for i, n in enumerate(target_nodes)}

    valid_df = df[df['source'].isin(source_nodes) & df['target'].isin(target_nodes)]
    if valid_df.empty:
        return sparse.csr_array((len(source_nodes), len(target_nodes)), dtype=np.float32), source_nodes, target_nodes

    src_indices = valid_df['source'].map(source_index).astype(int)
    tgt_indices = valid_df['target'].map(target_index).astype(int)
    data = np.ones(len(valid_df), dtype=np.float32)
    mat = sparse.csr_array(
        (data, (src_indices, tgt_indices)),
        shape=(len(source_nodes), len(target_nodes)),
        dtype=np.float32
    )

    invalid_sources = df[~df['source'].isin(source_nodes)]['source'].unique()
    invalid_targets = df[~df['target'].isin(target_nodes)]['target'].unique()
    if invalid_sources.size:
        print(f"Warning: {len(invalid_sources)} invalid {source_type} nodes")
    if invalid_targets.size:
        print(f"Warning: {len(invalid_targets)} invalid {target_type} nodes")

    return mat, source_nodes, target_nodes


def get_adjacency_tensor(edge_mats, relation_types, node_index):
    num_nodes = len(node_index)
    rel_index = {rel: i for i, rel in enumerate(relation_types)}
    tensors = [sparse.csr_array((num_nodes, num_nodes), dtype=np.float32) for _ in range(len(relation_types))]

    for rel, (mat, src_nodes, tgt_nodes) in edge_mats.items():
        if rel in relation_types:
            src_map = np.array([node_index.get(n, -1) for n in src_nodes], dtype=np.int32)
            tgt_map = np.array([node_index.get(n, -1) for n in tgt_nodes], dtype=np.int32)
            if not (src_nodes and tgt_nodes):
                continue
            mat = mat.tocoo()
            rows = src_map[mat.row]
            cols = tgt_map[mat.col]
            valid = (rows >= 0) & (cols >= 0)
            if valid.sum() > 0:
                tensors[rel_index[rel]] = sparse.csr_array(
                    (mat.data[valid], (rows[valid], cols[valid])),
                    shape=(num_nodes, num_nodes),
                    dtype=np.float32
                )

    return tensors, rel_index


def compute_similarity(mat, src_nodes, weight, local_index, num_entities):
    if not mat.nnz:
        return None
    src_idx = np.array([local_index[n] for n in src_nodes if n in local_index])
    if not src_idx.size:
        return None
    sim = mat @ mat.T
    sim_full = sparse.csr_array((num_entities, num_entities), dtype=np.float32)
    rows, cols = np.meshgrid(src_idx, src_idx, indexing='ij')
    sim_full[rows, cols] = sim.toarray()
    return sim_full * weight


def metapath_similarity(matrices, weights):
    if not matrices or not weights or len(matrices) != len(weights):
        raise ValueError("Matrices and weights must be non-empty and equal length")

    entities = set()
    for _, src_nodes, _ in matrices:
        entities.update(src_nodes)
    entities = sorted(list(entities))
    local_index = {n: i for i, n in enumerate(entities)}
    num_entities = len(entities)

    sims = Parallel(n_jobs=-1)(
        delayed(compute_similarity)(mat, src_nodes, weight, local_index, num_entities)
        for (mat, src_nodes, _), weight in zip(matrices, weights)
    )

    similarity = sparse.csr_array((num_entities, num_entities), dtype=np.float32)
    for sim in sims:
        if sim is not None:
            similarity += sim

    return similarity.toarray(), entities


def pad_tensor_to_shape(tensor, target_shape):
    if isinstance(tensor, sparse.csr_array):
        tensor = tensor.toarray()
    padded = np.zeros(target_shape, dtype=tensor.dtype)
    slices = tuple(slice(0, min(tensor.shape[i], target_shape[i])) for i in range(len(tensor.shape)))
    padded[slices] = tensor[slices]
    return padded


def save_metapaths(G, output_file, max_hops=3):
    compound_nodes = [n for n in G.nodes if n.startswith("Compound::")]
    disease_nodes = [n for n in G.nodes if n.startswith("Disease::")]
    metapaths = []

    for source in tqdm(compound_nodes, desc="Extracting metapaths"):
        for target in disease_nodes:
            for path in nx.all_simple_edge_paths(G, source=source, target=target, cutoff=max_hops):
                metaedges = [G.get_edge_data(u, v, key)['metaedge'] for u, v, key in path]
                nodenames = [source] + [v for _, v, _ in path]
                metapaths.append({
                    'Compound': source,
                    'Disease': target,
                    'Metapath': '->'.join(metaedges),
                    'Node_Path': '->'.join(nodenames)
                })

    df = pd.DataFrame(metapaths)
    df.to_csv(output_file, sep='\t', index=False)


def main():
    print("Loading node list...")
    nodes = load_node_list()
    all_nodes = sorted(list(nodes['Compound'].union(nodes['Disease'])))
    node_index = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(node_index)

    print(f"Number of compound nodes: {len(nodes['Compound'])}")
    print(f"Number of disease nodes: {len(nodes['Disease'])}")
    print(f"Number of gene nodes: {len(nodes['Gene'])}")
    print(f"Number of anatomy nodes: {len(nodes['Anatomy'])}")
    print(f"Number of symptom nodes: {len(nodes['Symptom'])}")
    print(f"Number of side effect nodes: {len(nodes['Side Effect'])}")
    print(f"Total nodes (compounds + diseases): {num_nodes}")

    print("Loading edge lists...")
    edges = load_edges()

    print("Building full graph...")
    G = build_graph(edges)

    edge_types = {
        'CuG': ('Compound', 'Gene'),
        'CbG': ('Compound', 'Gene'),
        'CdG': ('Compound', 'Gene'),
        'CcSE': ('Compound', 'Side Effect'),
        'CrC': ('Compound', 'Compound'),
        'CtD': ('Compound', 'Disease'),
        'CpD': ('Compound', 'Disease'),
        'DuG': ('Disease', 'Gene'),
        'DaG': ('Disease', 'Gene'),
        'DdG': ('Disease', 'Gene'),
        'DpS': ('Disease', 'Symptom'),
        'DlA': ('Disease', 'Anatomy'),
        'DrD': ('Disease', 'Disease')
    }

    adj_matrices = {}
    for edge_list in edge_types:
        if edge_list in edges:
            src_type, tgt_type = edge_types[edge_list]
            mat, src_nodes, tgt_nodes = create_adjacency_matrix(
                edges[edge_list],
                nodes[src_type],
                nodes[tgt_type],
                src_type,
                tgt_type
            )
            adj_matrices[edge_list] = (mat, src_nodes, tgt_nodes)
            print(f"Created adjacency matrix for {edge_list}: shape {mat.shape}")

    relation_types = ['CtD', 'CpD', 'Compound_Similarity', 'Disease_Similarity']

    print("Building compound similarity matrix...")
    compound_edge_lists = ['CuG', 'CbG', 'CdG', 'CcSE']
    compound_mats = [(adj_matrices[el][0], adj_matrices[el][1], adj_matrices[el][2]) for el in compound_edge_lists if
                     el in adj_matrices]
    weights = [0.25] * len(compound_mats)
    if not compound_mats:
        Ac = np.zeros((len(nodes['Compound']), len(nodes['Compound'])), dtype=np.float32)
        compound_entities = sorted(list(nodes['Compound']))
    else:
        Ac, compound_entities = metapath_similarity(compound_mats, weights)
    if 'CrC' in adj_matrices:
        cr_sim = adj_matrices['CrC'][0].toarray()
        cr_nodes = adj_matrices['CrC'][1]
        cr_map = {n: i for i, n in enumerate(cr_nodes)}
        Ac_full = np.zeros((len(compound_entities), len(compound_entities)), dtype=np.float32)
        if cr_nodes:
            cr_idx = [cr_map[n] for n in compound_entities if n in cr_map]
            Ac_full[np.ix_(cr_idx, cr_idx)] = cr_sim
        compound_sim = 0.5 * Ac + 0.5 * Ac_full
    else:
        compound_sim = Ac
    print(f"Compound Similarity Matrix: shape {compound_sim.shape}")

    print("Building disease similarity matrix...")
    disease_edge_lists = ['DuG', 'DaG', 'DdG', 'DpS', 'DlA']
    disease_mats = [(adj_matrices[el][0], adj_matrices[el][1], adj_matrices[el][2]) for el in disease_edge_lists if
                    el in adj_matrices]
    weights = [0.2] * len(disease_mats)
    if not disease_mats:
        Ad = np.zeros((len(nodes['Disease']), len(nodes['Disease'])), dtype=np.float32)
        disease_entities = sorted(list(nodes['Disease']))
    else:
        Ad, disease_entities = metapath_similarity(disease_mats, weights)
    if 'DrD' in adj_matrices:
        dr_sim = adj_matrices['DrD'][0].toarray()
        dr_nodes = adj_matrices['DrD'][1]
        dr_map = {n: i for i, n in enumerate(dr_nodes)}
        Ad_full = np.zeros((len(disease_entities), len(disease_entities)), dtype=np.float32)
        if dr_nodes:
            dr_idx = [dr_map[n] for n in disease_entities if n in dr_map]
            Ad_full[np.ix_(dr_idx, dr_idx)] = dr_sim
        disease_sim = 0.5 * Ad + 0.5 * Ad_full
    else:
        disease_sim = Ad
    print(f"Disease Similarity Matrix: shape {disease_sim.shape}")

    print("Building adjacency tensor...")
    tensor_edges = {
        rel: adj_matrices[rel] for rel in ['CtD', 'CpD'] if rel in adj_matrices
    }
    adj_tensors, rel_index = get_adjacency_tensor(tensor_edges, relation_types, node_index)

    print("Enriching heterogeneous network...")
    enriched_tensors = [sparse.csr_array((num_nodes, num_nodes), dtype=np.float32) for _ in range(len(relation_types))]
    enriched_tensors[rel_index['CtD']] = adj_tensors[rel_index['CtD']] if 'CtD' in rel_index else sparse.csr_array(
        (num_nodes, num_nodes))
    enriched_tensors[rel_index['CpD']] = adj_tensors[rel_index['CpD']] if 'CpD' in rel_index else sparse.csr_array(
        (num_nodes, num_nodes))

    compound_idx = np.array([i for i, n in enumerate(all_nodes) if n.startswith("Compound::")])
    disease_idx = np.array([i for i, n in enumerate(all_nodes) if n.startswith("Disease::")])
    comp_map = {n: i for i, n in enumerate(compound_entities)}
    dis_map = {n: i for i, n in enumerate(disease_entities)}
    comp_local_idx = np.array([comp_map.get(n, -1) for n in all_nodes], dtype=np.int32)
    dis_local_idx = np.array([dis_map.get(n, -1) for n in all_nodes], dtype=np.int32)
    valid_comp = comp_local_idx >= 0
    valid_dis = dis_local_idx >= 0
    comp_local_idx = comp_local_idx[valid_comp]
    dis_local_idx = dis_local_idx[valid_dis]

    if compound_idx.size and comp_local_idx.size:
        comp_sim_sub = sparse.csr_array(compound_sim[np.ix_(comp_local_idx, comp_local_idx)])
        rows, cols = np.meshgrid(compound_idx, compound_idx, indexing='ij')
        enriched_tensors[rel_index['Compound_Similarity']] = sparse.csr_array(
            (comp_sim_sub.toarray().ravel(), (rows.ravel(), cols.ravel())),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )

    if disease_idx.size and dis_local_idx.size:
        dis_sim_sub = sparse.csr_array(disease_sim[np.ix_(dis_local_idx, dis_local_idx)])
        rows, cols = np.meshgrid(disease_idx, disease_idx, indexing='ij')
        enriched_tensors[rel_index['Disease_Similarity']] = sparse.csr_array(
            (dis_sim_sub.toarray().ravel(), (rows.ravel(), cols.ravel())),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )

    for rel in ['CtD', 'CpD']:
        if rel in rel_index:
            rel_idx = rel_index[rel]
            Cd_mat = adj_tensors[rel_idx]
            enriched_Cd = sparse.lil_array((num_nodes, num_nodes), dtype=np.float32)
            if compound_idx.size and disease_idx.size and comp_local_idx.size and dis_local_idx.size:
                comp_sim_sub = sparse.csr_array(compound_sim[np.ix_(comp_local_idx, comp_local_idx)])
                dis_sim_sub = sparse.csr_array(disease_sim[np.ix_(dis_local_idx, dis_local_idx)])
                Cd_sub = Cd_mat[compound_idx[:, None], disease_idx]
                enriched_vals = (comp_sim_sub @ Cd_sub) + (Cd_sub @ dis_sim_sub.T)
                rows, cols = np.meshgrid(compound_idx, disease_idx, indexing='ij')
                enriched_Cd[rows, cols] = enriched_vals.toarray()
            enriched_tensors[rel_idx] = enriched_Cd.tocsr()

    enriched_tensor = np.stack([t.toarray() for t in enriched_tensors], axis=-1)
    np.save('enriched_Compound_Disease.npy', enriched_tensor)
    np.save('node_index.npy', np.array(all_nodes))

    print("Saving metapaths...")
    save_metapaths(G, 'compound_disease_metapaths.tsv', max_hops=3)

    print("File is saved.")


if __name__ == "__main__":
    main()