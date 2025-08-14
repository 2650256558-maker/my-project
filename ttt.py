import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch_geometric.utils import coalesce
from torch_geometric.data import Data
import itertools
import numpy as np

def construct_neighbor_matrix_with_self_loops_pyg(Q):
    # 获取 Q 的形状 (n, m)，其中 n 是像素数目，m 是超像素数目
    n, m = Q.shape

    # 初始化边列表
    edge_index = []

    # 遍历每一列
    for k in range(m):
        # 获取该列所有值为 1 的行索引
        rows_with_one = np.where(Q[:, k] == 1)[0]

        # 使用 itertools.combinations 生成所有像素对，避免嵌套循环
        for i, j in itertools.combinations(rows_with_one, 2):
            edge_index.append([i, j])
            edge_index.append([j, i])

    # 将边列表转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建稀疏矩阵时添加自环
    edge_index = torch.cat([edge_index, torch.arange(n).view(1, -1).repeat(2, 1)], dim=1)

    # 对边列表进行去重，确保没有重复的边
    edge_index, _ = coalesce(edge_index, None, n, n)

    # 将边索引转换为稀疏矩阵
    data = Data(edge_index=edge_index, edge_attr=None, num_nodes=n)

    return data


zero=torch.tensor(0.0).to(device)
one=torch.tensor(1.0).to(device)
alpha=torch.tensor(1e-8).to(device)


def Neigh_homo_ratio_sparse(adjacency, Y,DEVICE):
    """
    计算稀疏邻接矩阵 A 和标签向量 L 后输出 H。
    参数:
    A (Data): torch_geometric.Data 对象，其中包含 edge_index 和 edge_attr。
    L (torch.Tensor): 标签向量，形状为 (n,)。
    返回:
    torch.Tensor: 每个节点标签在其邻域中的出现次数除以该行的和。
    """
    adjacency = adjacency.to(DEVICE)
    Y = Y.to(DEVICE)

    # 将标签向量 L 转换为标签（加1确保标签从1开始）
    Y = Y + 1
    Y = Y.float()

    # 获取 edge_index 和 edge_attr
    edge_index = adjacency.edge_index.to(DEVICE)
    edge_attr = adjacency.edge_attr.to(DEVICE) if adjacency.edge_attr is not None else torch.ones(edge_index.size(1), device=DEVICE)

    # 计算每个节点的标签
    row_indices, col_indices = edge_index

    # 计算节点标签相同的邻居数
    label_same = (Y[row_indices] == Y[col_indices]).float()  # 创建一个向量，表示节点与邻居标签是否相同

    # 计算每个节点的度数（每个节点的邻居数量）
    A_num = torch.bincount(row_indices, minlength=Y.size(0)).float()

    # 使用稀疏矩阵来计算每个节点标签相同的邻居数量
    degree_label_same = torch.zeros_like(A_num)
    degree_label_same.index_add_(0, row_indices, label_same)

    # 计算每个节点标签在其邻域中的出现次数除以该行的和
    H = degree_label_same / A_num
    H = torch.where(A_num == 0, zero, H)

    return torch.mean(H)



def to_tensor(x):
    return torch.from_numpy(x.astype(np.float32)).to(device)

def ratio(gt,Q,DEVICE):
    gt = to_tensor(gt)
    h, w = gt.shape
    gt = gt.reshape([h * w, -1])
    gt = gt.view(-1)

    A = construct_neighbor_matrix_with_self_loops_pyg(Q)

    print(Neigh_homo_ratio_sparse(A, gt,DEVICE))