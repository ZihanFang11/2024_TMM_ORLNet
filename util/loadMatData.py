import os

import scipy.sparse as sp
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph



def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, ratio):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
            total_num -= 1
        else:
            p_unlabeled.append(idx)
    return p_labeled, p_unlabeled


def load_data_semi(args, dataset, normlize=True):
    data = sio.loadmat(args.path + dataset + '.mat')
    features = data['X']
    feature_list = []

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled, idx_unlabeled = generate_partition(labels=labels, ratio=args.ratio)
    labels = torch.from_numpy(labels).long()

    for i in range(features.shape[1]):
        if normlize:
            features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature_list.append(feature)
    return feature_list, labels, idx_labeled, idx_unlabeled

def load_data_single(args, dataset, device):
    data = sio.loadmat(args.path + dataset + '.mat')
    feature = data['X']

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    labels = torch.from_numpy(labels).long()

    feature = normalize(feature)

    return feature, labels

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_to_Lap(dataset,features, knns):
    # 遍历每个视图
    laps = []
    for i in range(len(features)):
        # 遍历每个视图
        direction_judge = os.getcwd() + '/Z_lap_matrix/' + dataset + '/' + 'v' + str(i) + '_knn' + str(
            knns) + '_lap.npz'
        if os.path.exists(direction_judge):
            print("Exist the Z laplacian matrix of " + str(i) + "th view of " + dataset)
            lap = ss.load_npz(direction_judge)
        else:
            print("Constructing the Z laplacian matrix of " + str(i) + "th view of " + dataset)
            # 返回该视图下每个样本距离最近的前n个样本
            temp = kneighbors_graph(features[i], knns)
            # 生成矩阵
            temp = sp.coo_matrix(temp)
            # build symmetric adjacency matrix
            temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
            lap = construct_laplacian(temp)
            save_direction = os.getcwd() + '/Z_lap_matrix/' + dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the Z laplacian matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) + '_knn' + str(knns) + '_lap.npz',lap)
        laps.append(lap.todense())

    return laps

def features_T_to_Lap(dataset,features_T, knn_G_ratio):

    laps = []
    for i in range(len(features_T)):
        knns = round(knn_G_ratio * features_T[i].shape[0])
        # 遍历每个视图
        direction_judge =  os.getcwd()+'/G_lap_matrix/' + dataset + '/' + 'v' + str(i) + '_knn' + str(knns) + '_lap.npz'
        if os.path.exists(direction_judge):
            print("Exist the G laplacian matrix of " + str(i) + "th view of " +dataset)
            lap = ss.load_npz(direction_judge)
        else:
            print("Constructing the G laplacian matrix of " + str(i) + "th view of " +dataset)

            # 返回该视图下每个样本距离最近的前n个样本
            # 生成矩阵
            temp = sp.coo_matrix(temp)
            # build symmetric adjacency matrix
            temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
            lap=construct_laplacian(temp)
            save_direction = os.getcwd()+'/G_lap_matrix/' + dataset + '/'
            if not os.path.exists(save_direction):
                os.makedirs(save_direction)
            print("Saving the G laplacian matrix to " + save_direction)
            ss.save_npz(save_direction + 'v' + str(i) +'_knn' + str(knns) + '_lap.npz', lap)
        laps.append(lap.todense())

    return laps



def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = sp.eye(adj.shape[0]) - adj_wave
    return lp
