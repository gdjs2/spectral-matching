import pdb
import time
import math
import torch
from torch import Tensor
from typing import Callable

def gen_affinity_matrix(adj_mat1: Tensor, adj_mat2: Tensor):
    n = adj_mat1.shape[0]
    m = adj_mat2.shape[0]
    
    values = []
    indices = []

    none_zero_indices1 = torch.nonzero(adj_mat1 == 1)
    none_zero_indices2 = torch.nonzero(adj_mat2 == 1)

    for i in none_zero_indices1:
        for j in none_zero_indices2:
            indices.append([i[0]*m + j[0], i[1]*m + j[1]])
            values.append(1)
            
    # print(torch.tensor(indices))
    # print(torch.tensor(values))
    
    return torch.sparse_coo_tensor(torch.tensor(indices).t(), torch.tensor(values), (n*m, n*m)).float()

def mat_quick_power(mat: Tensor, x: int):
    if x == 0:
        size = mat.shape[0]
        indicies = torch.arange(size).unsqueeze(0).repeat(2, 1)
        values = torch.ones(size)
        return torch.sparse_coo_tensor(indicies, values, (size, size)).float()
    tmp = mat_quick_power(mat, x>>1)
    tmp = torch.mm(tmp, tmp)
    if x & 1:
        tmp = torch.mm(tmp, mat)
    tmp = tmp / torch.norm(tmp)
    print(tmp.is_sparse)
    return tmp

def quick_power_iteration(affinity_matrix: Tensor, max_iter: int, b_k):
    mat_p = mat_quick_power(affinity_matrix, max_iter)
    b_k1 = torch.mm(mat_p, b_k)
    b_k = b_k1 / torch.norm(b_k1)
    return b_k
    

def power_iteration(affiniry_matrix: Tensor, max_iter: int, b_k):
    for _ in range(max_iter):
        b_k1 = torch.mm(affiniry_matrix, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k

def spectral_clustering(affiniry_matrix: Tensor, max_iter: int, n: int, m: int, select: Callable[[Tensor], Tensor], b_k):
    eigenvector = power_iteration(affiniry_matrix, max_iter, b_k)
    print(eigenvector)
    return select(eigenvector, n, m)

def quick_spectral_clustering(affiniry_matrix: Tensor, max_iter: int, n: int, m: int, select: Callable[[Tensor], Tensor], b_k):
    eigenvector = quick_power_iteration(affiniry_matrix, max_iter, b_k)
    print(eigenvector)
    return select(eigenvector, n, m)

def binary_select(eigenvector: Tensor, n: int, m: int):
    x_star = eigenvector
    selected = []
    while True:
        max_index = torch.argmax(x_star).item()
        if x_star[max_index] == 0:
            return selected
        selected.append((max_index // m, max_index % m))
        selected_i, selected_j = max_index // m, max_index % m
        deleted = [selected_i*m + j for j in range(m)] + [i*m + selected_j for i in range(n)]
        # pdb.set_trace()
        x_star[deleted] = 0
    
if __name__ == '__main__':
    
    adj_mat1 = torch.tensor([[0, 1, 0, 0],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0]])
    
    adj_mat2 = torch.tensor([[0, 1, 1, 0],
                             [1, 0, 0, 0],
                             [1, 0, 0, 1],
                             [0, 0, 1, 0]])
    
    affinity_mat = gen_affinity_matrix(adj_mat1, adj_mat2)

    b_k = torch.rand((affinity_mat.shape[0], 1))
    max_iter = 100000

    start_time = time.time()
    # vec = spectral_clustering(affinity_mat, max_iter, 4, 4, binary_select, b_k)
    # print(vec, ", time: ", time.time() - start_time)

    start_time = time.time()
    vec = quick_spectral_clustering(affinity_mat, max_iter, 4, 4, binary_select, b_k)
    print(vec, ", time: ", time.time() - start_time)
