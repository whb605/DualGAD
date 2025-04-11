import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as fn


def node_iter(G):
    if float(nx.__version__[:3]) < 2.0:
        return G.nodes()
    else:
        return G.nodes


def node_dict(G):
    if float(nx.__version__[:3]) > 2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def f1_score_all(y_pred, y_true, sign=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if sign == 1:
        total_anomalies = int(np.sum(y_true))
        anomaly_indices = np.argsort(y_pred)[-total_anomalies:]
    else:
        total_anomalies = int(np.sum(1 - y_true))
        anomaly_indices = np.argsort(-y_pred)[-total_anomalies:]
    if sign == 1:
        y_pred_label = np.zeros_like(y_pred)
    else:
        y_pred_label = np.ones_like(y_pred)
    y_pred_label[anomaly_indices] = sign

    TP = np.sum((y_pred_label == sign) & (y_true == sign))
    FP = np.sum((y_pred_label == sign) & (y_true == 1 - sign))
    FN = np.sum((y_pred_label == 1 - sign) & (y_true == sign))
    if TP == 0: return 0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_reconstruct_diff(X, X_pred, A, A_pred):
    A1 = torch.where(A > 0, torch.tensor(1.0, device="cuda"), A)
    adj_diff = fn.binary_cross_entropy(A_pred, A1, reduction='none')
    adj_diff = adj_diff.mean(dim=-1)

    att_diff = fn.mse_loss(X_pred, X, reduction='none')
    att_diff = att_diff.mean(dim=-1)
    horizontal_concat = torch.cat((adj_diff, att_diff), dim=1)
    return horizontal_concat


def align_lists(a, b):
    if len(a) > len(b):
        repeat_count = (len(a) // len(b)) + 1
        b *= repeat_count
        b = b[:len(a)]
    elif len(a) < len(b):
        repeat_count = (len(b) // len(a)) + 1
        a *= repeat_count
        a = a[:len(b)]
    return a, b
