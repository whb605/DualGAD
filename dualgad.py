import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearEncoder(nn.Module):

    def __init__(self, hidden_dim, node_num):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * node_num, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).cuda()

    def forward(self, Z):
        score = self.linear(torch.reshape(Z, (len(Z), -1)))
        return score


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.transpose(1, 2)))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphEncoder(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, node_num):
        super(GraphEncoder, self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, activation=F.relu)
        self.hidd_gcn = GraphConvSparse(hidden1_dim, hidden2_dim, activation=torch.sigmoid)

    def forward(self, X, adj):
        hidden = self.base_gcn(X, adj)
        sampled_z = self.hidd_gcn(hidden, adj)
        return sampled_z


class GraphDecoder(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim):
        super(GraphDecoder, self).__init__()
        self.gcn_decoder1 = nn.Linear(hidden2_dim, hidden1_dim, bias=True)
        self.gcn_decoder2 = nn.Linear(hidden1_dim, input_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, X, adj):
        x = self.relu(self.gcn_decoder1(torch.matmul(adj, X)))
        x = self.gcn_decoder2(torch.matmul(adj, x))
        A = dot_product_decode(X)
        return x, A


class GraphGrade(nn.Module):

    def __init__(self, input_dim, node_num):
        super(GraphGrade, self).__init__()
        self.gcn_grade = nn.Linear(input_dim * node_num, 1)
        self.act = nn.Sigmoid()

    def forward(self, X):
        x = self.act(self.gcn_grade(X))
        return x
