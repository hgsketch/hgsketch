import os
import argparse
import torch
import numpy as np
import time
from hgsketch_utils import hgsketch, classifier, load_dataset, save_embedding
from scipy.spatial.distance import pdist, squareform

def main():
    parser = argparse.ArgumentParser(description="Run heterogeneous graph classification with specified parameters.")
    parser.add_argument('--ds_name', type=str, required=True, help='Dataset name (Cuneiform, sr_ARE, DBLP, nr_BIO)')
    parser.add_argument('--R', type=int, required=True, help='Iterations')
    parser.add_argument('--K', type=int, required=True, help='Maximum dimension of simplexes')
    args = parser.parse_args()

    if args.ds_name == 'Cuneiform':
        task_type = 'multiclass'
        lr_params = {'max_iter': 20000, 'solver': 'liblinear', 'C': 1}
        svc_C = 1.0
    elif args.ds_name == 'sr_ARE':
        task_type = 'binary'
        lr_params = {'max_iter': 20000, 'solver': 'liblinear', 'C': 0.02}
        svc_C = 0.5
    elif args.ds_name == 'DBLP':
        task_type = 'binary'
        lr_params = {'max_iter': 30000, 'solver': 'lbfgs', 'C': 0.004}
        svc_C = 0.5
    elif args.ds_name == 'nr_BIO':
        task_type = 'binary'
        lr_params = {'max_iter': 30000, 'solver': 'lbfgs', 'C': 0.02}
        svc_C = 0.5
    else:
        raise ValueError(f"Unsupported dataset: {args.ds_name}")

    dataset = load_dataset(args.ds_name)

    S_List = [20, 30, 35]
    D = 25

    labels = []
    after_sketch_matrix = np.zeros((len(dataset), D * (sum(S_List[:args.K + 1]))), dtype=np.uint8)

    start_time = time.time()
    for g in range(len(dataset)):
        data = dataset[g]
        status = hgsketch(g, data.edge_index, data.x, args.K, D, args.R, S_List, after_sketch_matrix)
        labels.append(int(data.y))
    emb_result = after_sketch_matrix
    end_emb_time = time.time()
    print(f"Embedding Time: {end_emb_time - start_time:.2f}")

    # save_embedding(ds_name, emb_result)

    # Logistic Regression
    start_expand_time = time.time()
    deal_emb = np.concatenate((emb_result, 1 - emb_result), axis=1)
    time_tmp = time.time()

    start_lr_time = time.time()
    lr_result = classifier(
        deal_emb, np.array(labels), task_type=task_type, classifier_type='lr', lr_params=lr_params
    )
    end_lr_time = time.time()

    total_lr_time = (end_emb_time - start_time) + (time_tmp - start_expand_time) + (end_lr_time - start_lr_time)
    print(f"Runtime (LR): {total_lr_time:.2f}")

    # SVM
    start_kernel_time = time.time()
    gram_matrix = args.R * (1 - squareform(pdist(emb_result, 'hamming')))
    end_kernel_time = time.time()
    sum_kernel_time = end_kernel_time - start_kernel_time

    start_svm_time = time.time()
    svm_result = classifier(
        gram_matrix, np.array(labels), task_type=task_type, classifier_type='svm', svc_C=svc_C
    )
    end_svm_time = time.time()

    total_svm_time = (end_emb_time - start_time) + (end_svm_time - start_svm_time) + sum_kernel_time
    print(f"Runtime (SVM): {total_svm_time:.2f}")

if __name__ == "__main__":
    main()