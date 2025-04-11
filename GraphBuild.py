import networkx as nx
import numpy as np
import torch.utils.data
from util import *
import util


class GraphBuild(torch.utils.data.Dataset):
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        super(GraphBuild, self).__init__()
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []

        self.assign_feat_all = []
        self.max_num_nodes = max_num_nodes

        if features == 'default':
            self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            feat = ...
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i, u in enumerate(G.nodes()):
                    f[i, :] = util.node_dict(G)[u]['feat']
                feat = f

            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                if self.max_num_nodes > G.number_of_nodes():
                    degs = np.expand_dims(
                        np.pad(degs, (0, self.max_num_nodes - G.number_of_nodes()), 'constant', constant_values=0),
                        axis=1)
                elif self.max_num_nodes < G.number_of_nodes():
                    deg_index = np.argsort(degs, axis=0)
                    deg_ind = deg_index[0: G.number_of_nodes() - self.max_num_nodes]
                    degs = np.delete(degs, [deg_ind], axis=0)
                    degs = np.expand_dims(degs, axis=1)
                else:
                    degs = np.expand_dims(degs, axis=1)
                feat = degs
            num_nodes = adj.shape[0]
            if self.max_num_nodes > num_nodes:
                adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
                adj_padded[:num_nodes, :num_nodes] = adj
            elif self.max_num_nodes < num_nodes:
                degs = np.sum(np.array(adj), 1)
                deg_index = np.argsort(degs, axis=0)
                deg_ind = deg_index[0:num_nodes - self.max_num_nodes]
                adj_padded = np.delete(adj, [deg_ind], axis=0)
                adj_padded = np.delete(adj_padded, [deg_ind], axis=1)
            else:
                adj_padded = adj

            adj_label = adj_padded + sp.eye(adj_padded.shape[0])
            adj = adj_label.tolist()
            adj = normalize_adj(adj)
            adj = adj.toarray()
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            self.feature_all.append(feat)
            self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]
        self.adj_all = torch.tensor(np.array(self.adj_all), dtype=torch.float32).cuda()
        self.feature_all = torch.tensor(np.array(self.feature_all), dtype=torch.float32).cuda()
        self.label_all = torch.tensor(np.array(self.label_all), dtype=torch.float32).cuda()

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        return {'adj': adj,
                'feats': self.feature_all[idx],
                'label': self.label_all[idx]
                }