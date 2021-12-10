import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    W=W.to(device=torch.device('cpu'))
    assert W.shape[0] == W.shape[1]
    W=W.cpu().detach().numpy()

    N = W.shape[0]
    D = np.zeros([N, N], dtype=type(W[0][0]))


    W = W + np.identity(N) # 为邻居矩阵加上自连接
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i][j] != 0.:
                D[i][j] = 1
    # print(D)
    D = np.diag(np.sum(D, axis=1))
    # print("D:",D)
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    # print("*****")
    # print(sym_norm_Adj_matrix.device)
    # print(D.device)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))
    # N = W.shape[0]
    # W = W + torch.from_numpy(np.identity(N)) # 为邻居矩阵加上自连接
    # # D = np.diag(np.sum(W,axis=1))
    # D = torch.diag(torch.sum(W,dim=1))
    # sym_norm_Adj_matrix = torch.dot(np.sqrt(D),W)
    # sym_norm_Adj_matrix = torch.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32)
        self.sym_norm_Adj_matrix = F.softmax(self.sym_norm_Adj_matrix, dim=1)
        self.alpha = nn.Parameter(torch.FloatTensor([0.4]), requires_grad=True)  # D
        self.beta = nn.Parameter(torch.FloatTensor([0.6]), requires_grad=True)  # S


        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        # print(x_gconv.shape)
        # static_out=torch.einsum("nn,bnc->bnc",self.sym_norm_Adj_matrix,x)
        # print(static_out.shape)
        # gcn_out=self.alpha*static_out+self.beta*x_gconv
        gcn_out=x_gconv
        return gcn_out