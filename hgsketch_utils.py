import numpy as np
import gudhi as gd
from scipy.sparse import coo_matrix
import itertools
import os
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import label_binarize
from itertools import combinations
import random
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import zipfile

def load_dataset(ds_name, data_dir='./Dataset'):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist!")
    zip_path = os.path.join(data_dir, f'{ds_name}.zip')
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset .zip file '{zip_path}' does not exist!")
    extract_dir = os.path.join(data_dir, ds_name)
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Dataset '{ds_name}' extracted to '{extract_dir}'.")
    except zipfile.BadZipFile:
        raise RuntimeError(f"Failed to extract '{zip_path}': File is not a valid .zip file.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract '{zip_path}': {str(e)}")

    dataset_path = os.path.join(extract_dir, f'{ds_name}.pth')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' does not exist after extraction!")
    try:
        dataset = torch.load(dataset_path)
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from '{dataset_path}': {str(e)}")


def save_embedding(ds_name, embedding):
    folder_path = os.path.join('saved_embedding', ds_name)
    os.makedirs(folder_path, exist_ok=True)
    file_name = f'{ds_name}_embedding.npy'
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, embedding)


def classifier(X, y, task_type='multiclass', classifier_type='lr', lr_params=None, svc_C=1.0):
    accuracy = []
    roc_auc = []
    random_seed = random.randrange(0, 1000000)
    print(f"Random seed: {random_seed}")
    kf = KFold(n_splits=5, random_state=random_seed, shuffle=True)

    if task_type == 'multiclass':
        y_bin = label_binarize(y, classes=np.unique(y))
        n_classes = y_bin.shape[1]

    for train_index, test_index in kf.split(X):
        if classifier_type == 'lr':
            if lr_params is None:
                lr_params = {'max_iter': 20000, 'solver': 'liblinear', 'C': 1}
            clf = OneVsRestClassifier(LogisticRegression(**lr_params))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        elif classifier_type == 'svm':
            X_train = X[train_index][:, train_index]
            X_test = X[test_index][:, train_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = svm.SVC(C=svc_C, kernel='precomputed', probability=True)
        else:
            raise ValueError("Invalid classifier_type. Choose 'lr' or 'svm'.")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy.append(acc)

        if task_type == 'multiclass':
            y_test_bin = y_bin[test_index]
            roc_auc_val = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='micro')
        elif task_type == 'binary':
            if len(np.unique(y_test)) == 2:
                roc_auc_val = roc_auc_score(y_test, y_pred_prob[:, 1])
            else:
                roc_auc_val = np.nan
        else:
            raise ValueError("Invalid task_type. Choose 'multiclass' or 'binary'.")

        roc_auc.append(roc_auc_val)

    print(f'Accuracy: {np.mean(accuracy):.4f} ---- ROC-AUC: {np.mean(roc_auc):.4f}')

    return accuracy, roc_auc

def simplex_extraction(edges,node_nums,K):
    simplex_set=gd.SimplexTree()
    for v in range(node_nums):
        simplex_set.insert([v])
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        simplex_set.insert(edge)
        simplex_set.expansion(K)
    return simplex_set

def simplex_data(simplex_set, node_nums):
    complex_dim = simplex_set.dimension()
    simplex_lists = [[] for _ in range(complex_dim + 1)]
    simplex_to_index = [{} for _ in range(complex_dim + 1)]
    simplex_lists[0] = [[v] for v in range(node_nums)]
    simplex_to_index[0] = {tuple([v]): v for v in range(node_nums)}
    for simplex, _ in simplex_set.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue
        next_id = len(simplex_lists[dim])
        simplex_to_index[dim][tuple(simplex)] = next_id
        simplex_lists[dim].append(simplex)
    return simplex_lists, simplex_to_index

def build_boundary_matrices(simplex_set, id_maps):
    boundary_matrices = []
    complex_dim = simplex_set.dimension()
    for dim in range(1, complex_dim + 1):
        simplex_id = []
        boundary_id = []
        values = []
        for simplex in simplex_set.get_simplices():
            simplex_tuple = tuple(simplex[0])
            simplex_dim = len(simplex_tuple) - 1
            if simplex_dim == dim:
                boundaries = combinations(simplex_tuple, simplex_dim)
                for j, boundary in enumerate(boundaries):
                    values.append((-1) ** j)
                    simplex_id.append(id_maps[dim][simplex_tuple])
                    boundary_id.append(id_maps[dim - 1][tuple(boundary)])
        boundary_matrix = coo_matrix(
            (values, (boundary_id, simplex_id)),
            shape=(len(id_maps[dim - 1]), len(id_maps[dim]))
        ).toarray()
        boundary_matrices.append(boundary_matrix)
    return boundary_matrices

def build_hodge_laplacian(boundary_matrices):
    laplacians = []
    up = np.array(boundary_matrices[0] @ boundary_matrices[0].T)
    laplacians.append(up)
    for d in range(len(boundary_matrices) - 1):
        down = np.array(boundary_matrices[d].T @ boundary_matrices[d])
        up = np.array(boundary_matrices[d + 1] @ boundary_matrices[d + 1].T)
        laplacians.append(down + up)
    down = np.array(boundary_matrices[-1].T @ boundary_matrices[-1])
    laplacians.append(down)
    return laplacians

def initialization(hg_fea, simplex_tables,id_maps,hg_fea_dim):
    hg_features = list()
    hg_features.append(hg_fea.numpy())
    for dim in range(1,len(simplex_tables)):
       fea_d = np.zeros((len(simplex_tables[dim]), hg_fea_dim))
       for c, cell in enumerate(simplex_tables[dim]):
                for _, node in enumerate(cell):
                  fea_d[id_maps[dim][tuple(cell)]] = np.maximum(fea_d[id_maps[dim][tuple(cell)]],hg_fea[int(node)])
       hg_features.append(fea_d)
    return hg_features

def sgn(res):
    return res > 0

def hgsketch(graph_id, edge_index, hg_fea, K, D, R, S_List, after_sketch):
    np.random.seed(222)
    node_nums = hg_fea.shape[0]
    hg_fea_dim = hg_fea.shape[1]
    simplex_set = simplex_extraction(edge_index, node_nums, K)
    simplex_lists, simplex_to_index = simplex_data(simplex_set, node_nums)
    boundary_matrices = build_boundary_matrices(simplex_set, simplex_to_index)
    if len(boundary_matrices) == 0:
        return False
    hodge_laplacians = build_hodge_laplacian(boundary_matrices)
    simplex_dim = len(simplex_lists) if K != 0 else 1

    M = [[] for _ in range(simplex_dim)]
    N = [[] for _ in range(simplex_dim)]
    for k in range(simplex_dim):  # Local amplification operator
        M[k] = hodge_laplacians[k] * hodge_laplacians[k]
    for k in range(simplex_dim):  # Global enhancement operator
        N[k] = M[k] @ M[k]

    Hin = initialization(hg_fea, simplex_lists, simplex_to_index, hg_fea_dim)

    for k in range(simplex_dim):
        Hin[k] = N[k] @ Hin[k]
    Hout = []
    for i in range(R):
        for k in range(simplex_dim):
            W = np.random.randn(Hin[k].shape[1], D)
            Hin[k] = sgn(Hin[k] @ W)
        for k in range(simplex_dim):
            if Hin[k].shape[0] < S_List[k]:
                padding = np.zeros((S_List[k] - Hin[k].shape[0], Hin[k].shape[1]))
                Hout.append(np.concatenate((Hin[k], padding), 0))
            else:
                Hout.append(Hin[k][:S_List[k], :])
            flattened_array = Hout[k].flatten()
            if k == 0:
                output = flattened_array
            else:
                output = np.concatenate((output, flattened_array), 0)
    after_sketch[graph_id, :len(output)] = output

    return True
