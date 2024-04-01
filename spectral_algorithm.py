import time
import torch
import logging
import pygmtools as gm
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

def quick_power_iteration(affinity_matrix: Tensor, max_iter: int, b_k, eps = 1e-6):
    affinity_mat_power = affinity_matrix
    for iter_num in range(max_iter):
        b_k_old = b_k
        
        b_k1 = torch.mm(affinity_mat_power, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        if torch.norm(b_k - b_k_old) < eps:
            logging.debug(f"Quick PI Converged at iteration {iter_num}")
            break

        affinity_mat_power = torch.mm(affinity_mat_power, affinity_mat_power)
        affinity_mat_power /= torch.norm(affinity_mat_power)
    
    if iter_num == max_iter - 1:
        logging.debug(f"Quick PI did not converge at iteration {max_iter}")
    
    return b_k

def power_iteration(affiniry_matrix: Tensor, max_iter: int, b_k, eps = 1e-6):
    for iter_num in range(max_iter):
        b_k_old = b_k
        b_k1 = torch.mm(affiniry_matrix, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        logging.debug(b_k)
        if torch.norm(b_k - b_k_old) < eps:
            logging.debug(f"PI Converged at iteration {iter_num}")
            break

    if iter_num == max_iter - 1:
        logging.debug(f"PI did not converge at iteration {max_iter}")
    return b_k

def spectral_clustering(affiniry_matrix: Tensor, max_iter: int, n: int, m: int, select: Callable[[Tensor], Tensor], b_k: Tensor = None):
    eigenvector = power_iteration(affiniry_matrix, max_iter, b_k)
    logging.debug(f"Eigen Vector: {eigenvector}")
    return select(eigenvector, n, m)

def quick_spectral_clustering(affiniry_matrix: Tensor, max_iter: int, n: int, m: int, select: Callable[[Tensor], Tensor], b_k: Tensor = None):
    eigenvector = quick_power_iteration(affiniry_matrix, max_iter, b_k)
    logging.debug(f"Eigen Vector: {eigenvector}")
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
    return selected

def hungarian_select(eigenvector: Tensor, n: int, m: int):
    res = gm.hungarian(eigenvector.reshape(n, m), nproc=3, backend="pytorch").nonzero().tolist()
    return res

if __name__ == '__main__':
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d][%(levelname)s]: %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S'
    )
    
    adj_mat1 = torch.tensor([[0, 1, 0, 0],
                             [1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0]])
    
    adj_mat2 = torch.tensor([[0, 1, 1, 0],
                             [1, 0, 0, 0],
                             [1, 0, 0, 1],
                             [0, 0, 1, 0]])
    
    affinity_mat = gen_affinity_matrix(adj_mat1, adj_mat2)
    logging.debug(f"Affinity Matrix: {affinity_mat}")

    b_k = torch.rand((affinity_mat.shape[0], 1))
    max_iter = 100000

    # start_time = time.time()
    # vec = spectral_clustering(affinity_mat, max_iter, 4, 4, binary_select, b_k)
    # logging.debug(f"Time for normal clustering: {time.time() - start_time}")
    # logging.debug(f"Selected: {vec}")
    start_time = time.time()
    vec = quick_spectral_clustering(affinity_mat, max_iter, 4, 4, binary_select, b_k)
    logging.debug(f"Time for quick clustering: {time.time() - start_time}")
    logging.debug(f"Selected: {vec}")
    
    start_time = time.time()
    vec = quick_spectral_clustering(affinity_mat, max_iter, 4, 4, hungarian_select, b_k)
    logging.debug(f"Time for quick clustering: {time.time() - start_time}")
    logging.debug(f"Selected: {vec}")
