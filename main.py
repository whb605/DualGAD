import torch.utils.data
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score
import argparse
import load_data
from GraphBuild import GraphBuild
import random
from sklearn.model_selection import StratifiedKFold
from dualgad import *
from util import *


def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default='AIDS', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim1', dest='hidden_dim1', default=128, type=int, help='Hidden dimension1')
    parser.add_argument('--hidden-dim2', dest='hidden_dim2', default=64, type=int, help='Hidden dimension2')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature')
    # parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--sign', dest='sign', type=int, default=0, help='sign of graph anomaly')
    parser.add_argument('--ratio', dest='ratio', type=float, default=0.2, help='hyper-parameter')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_adj_loss(A_preds, adj_labels, weight_tensors, norms):
    length = A_preds.shape[0]
    loss = 0
    try:
        for i in range(length):
            A_pred = A_preds[i]
            adj_label = adj_labels[i]
            weight_tensor = weight_tensors[i]
            norm = norms[i]
            loss += norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)
    except:
        loss = 0
    return loss / length


def train(data_test_loader, data_abnormal_loader, data_normal_loader, encoder_n, encoder_a, decoder, grade_n, grade_a,
          args):
    parameters_all = (list(encoder_n.parameters()) +
                      list(encoder_a.parameters()) +
                      list(decoder.parameters()) +
                      list(grade_n.parameters()) +
                      list(grade_a.parameters()))
    optimizerG = torch.optim.Adam(parameters_all, lr=args.lr)
    auroc_final = 0
    max_AUC = 0
    dataloader_iter_n = iter(data_abnormal_loader)
    encoder_n.train(), encoder_a.train(), decoder.train()

    for epoch in range(args.num_epochs):
        loss_s, loss_scores, loss_total = 0, 0, 0
        for batch_idx_p, data_p in enumerate(data_normal_loader):
            adj_p = data_p['adj']
            h0_p = data_p['feats']
            try:
                data_n = next(dataloader_iter_n)
            except StopIteration:
                dataloader_iter_n = iter(data_abnormal_loader)
                data_n = next(dataloader_iter_n)
            adj_n = data_n['adj']
            h0_n = data_n['feats']

            p_nor_embedding = encoder_n(h0_p, adj_p)
            p_ano_embedding = encoder_a(h0_p, adj_p)
            n_nor_embedding = encoder_n(h0_n, adj_n)
            n_ano_embedding = encoder_a(h0_n, adj_n)

            loss_similarity_p = torch.exp(-F.mse_loss(p_nor_embedding, p_ano_embedding))
            loss_similarity_n = torch.exp(-F.mse_loss(n_nor_embedding, n_ano_embedding))
            loss_s += loss_similarity_p.item()
            loss_s += loss_similarity_n.item()
            loss_similarity = (loss_similarity_p + loss_similarity_n) / 2

            p_nor_X, p_nor_A = decoder(p_nor_embedding, adj_p)
            p_ano_X, p_ano_A = decoder(p_ano_embedding, adj_p)
            n_nor_X, n_nor_A = decoder(n_nor_embedding, adj_n)
            n_ano_X, n_ano_A = decoder(n_ano_embedding, adj_n)

            p_nor_diff = get_reconstruct_diff(h0_p, p_nor_X, adj_p, p_nor_A)
            p_ano_diff = get_reconstruct_diff(h0_p, p_ano_X, adj_p, p_ano_A)
            n_nor_diff = get_reconstruct_diff(h0_n, n_nor_X, adj_n, n_nor_A)
            n_ano_diff = get_reconstruct_diff(h0_n, n_ano_X, adj_n, n_ano_A)

            p_gn = grade_n(p_nor_diff)
            p_ga = grade_a(p_ano_diff)
            n_gn = grade_n(n_nor_diff)
            n_ga = grade_a(n_ano_diff)

            scores = torch.cat([p_gn, p_ga, n_gn, n_ga], dim=0)
            scores_label = torch.cat([
                torch.zeros_like(p_gn),
                torch.ones_like(p_ga),
                torch.ones_like(n_gn),
                torch.zeros_like(n_ga)
            ], dim=0)
            loss_score = F.binary_cross_entropy(scores, scores_label)
            loss_scores += loss_score.item()
            loss = loss_score * (1 - args.ratio) + loss_similarity * args.ratio
            loss_total += loss
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

        if (epoch + 1) % 5 == 0 and epoch > 0:
            encoder_n.eval(), encoder_a.eval(), decoder.eval(), grade_n.eval(), grade_a.eval()
            scores = []
            n_scores = []
            a_scores = []
            labels = []

            for batch_idx, data in enumerate(data_test_loader):
                adj = data['adj']
                feat = data['feats']
                label = data['label']

                n_emd = encoder_n(feat, adj)
                a_emd = encoder_a(feat, adj)
                n_X, n_A = decoder(n_emd, adj)
                a_X, a_A = decoder(a_emd, adj)
                n_recon_error = grade_n(get_reconstruct_diff(feat, n_X, adj, n_A))
                a_recon_error = grade_a(get_reconstruct_diff(feat, a_X, adj, a_A))

                if args.sign == 1:
                    score = n_recon_error - a_recon_error
                else:
                    score = a_recon_error - n_recon_error

                scores.extend(score.squeeze().tolist())
                n_scores.extend(n_recon_error.squeeze().tolist())
                a_scores.extend((-a_recon_error).squeeze().tolist())
                labels.extend(label.squeeze().tolist())

            target_class = 1
            binary_labels = [1 if label == target_class else 0 for label in labels]

            fpr_ab, tpr_ab, thr_ = roc_curve(binary_labels, scores)
            test_roc_ab = auc(fpr_ab, tpr_ab)
            n_auc = roc_auc_score(binary_labels, n_scores)
            a_auc = roc_auc_score(binary_labels, a_scores)
            print('abnormal detection: auroc_ab: {}, 总loss: {} loss_similarity: {} loss_score:{}, n_auc: {}, '
                  'a_auc: {}'.format(format(test_roc_ab, '.4f'),
                                     format(loss_total / epoch, '.4f'),
                                     format(loss_s / epoch, '.4f'),
                                     format(loss_scores / epoch, '.4f'),
                                     format(n_auc, '.4f'),
                                     format(a_auc, '.4f')))
            if test_roc_ab > max_AUC:
                max_AUC = test_roc_ab
        if epoch == (args.num_epochs - 1):
            auroc_final = max_AUC
            best_fpr = fpr_ab
            best_tpr = tpr_ab
            F1_Score = f1_score_all(scores, binary_labels, args.sign)
            best_precision, best_recall, _ = precision_recall_curve(binary_labels, scores)

    return auroc_final, F1_Score, best_fpr, best_tpr, best_precision, best_recall


if __name__ == '__main__':
    args = arg_parse()
    n_k = 5
    setup_seed(args.seed)
    large = not args.DS.find("Tox21_") == -1

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print('GraphNumber: {}'.format(datanum))
    graphs_label = [graph.graph['label'] for graph in graphs]

    if large:
        DST = args.DS[:args.DS.rfind('_')] + "_testing"
        graphs_testgroup = load_data.read_graphfile(args.datadir, DST, max_nodes=args.max_nodes)
        datanum_test = len(graphs_testgroup)
        if args.max_nodes == 0:
            max_nodes_num = max([max([G.number_of_nodes() for G in graphs_testgroup]), max_nodes_num])
        else:
            max_nodes_num = args.max_nodes
        graphs_label_test = [graph.graph['label'] for graph in graphs_testgroup]

        graphs_all = graphs + graphs_testgroup
        graphs_label_all = graphs_label + graphs_label_test
    else:
        graphs_all = graphs
        graphs_label_all = graphs_label

    kfd = StratifiedKFold(n_splits=n_k, random_state=args.seed, shuffle=True)
    result_auc = []
    result_f1 = []

    for k, (train_index, test_index) in enumerate(kfd.split(graphs_all, graphs_label_all)):
        print(k + 1, '次实验结果')
        graphs_train = [graphs_all[i] for i in train_index]
        graphs_test = [graphs_all[i] for i in test_index]
        graphs_normal = []
        graphs_abnormal = []

        for graph in graphs_train:
            if graph.graph['label'] == args.sign:
                graphs_abnormal.append(graph)
            else:
                graphs_normal.append(graph)

        num_normal = len(graphs_normal)
        num_abnormal = len(graphs_abnormal)
        num_test = len(graphs_test)
        graphs_normal, graphs_abnormal = align_lists(graphs_normal, graphs_abnormal)

        dataset_test = GraphBuild(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=args.batch_size)

        dataset_normal = GraphBuild(graphs_normal, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_normal_loader = torch.utils.data.DataLoader(dataset_normal, shuffle=True, batch_size=args.batch_size)

        dataset_abnormal = GraphBuild(graphs_abnormal, features=args.feature, normalize=False,
                                      max_num_nodes=max_nodes_num)
        data_abnormal_loader = torch.utils.data.DataLoader(dataset_abnormal, shuffle=True, batch_size=args.batch_size)

        feat_dim, hidden_dim1, hidden_dim2 = dataset_test.feat_dim, args.hidden_dim1, args.hidden_dim2

        encoder_n = GraphEncoder(feat_dim, hidden_dim1, hidden_dim2, max_nodes_num).cuda()
        encoder_a = GraphEncoder(feat_dim, hidden_dim1, hidden_dim2, max_nodes_num).cuda()
        decoder = GraphDecoder(feat_dim, hidden_dim1, hidden_dim2).cuda()
        grade_n = GraphGrade(2, max_nodes_num).cuda()
        grade_a = GraphGrade(2, max_nodes_num).cuda()

        result, F1_Score, best_fpr, best_tpr, best_precision, best_recall = \
            train(data_test_loader, data_abnormal_loader, data_normal_loader, encoder_n, encoder_a, decoder, grade_n,
                  grade_a, args)
        result_auc.append(result)
        result_f1.append(F1_Score)

    result_auc = np.array(result_auc)
    result_f1 = np.array(result_f1)
    auc_avg = np.mean(result_auc)
    f1_avg = np.mean(result_f1)
    auc_std = np.std(result_auc)
    f1_std = np.std(result_f1)
    print("数据集：{} auc_avg：{} auc_std：{}".format(args.DS, auc_avg, auc_std))
