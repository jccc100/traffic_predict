import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

device=torch.device('cuda')
# device=torch.device('cpu')

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
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
        # print(self.weights_pool.shape)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # print(weights.shape)
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # print(x_g.shape)

        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

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


    W = W + 0.5*np.identity(N) # 为邻居矩阵加上自连接
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if W[i][j] != 0.:
                D[i][j] = 1
    # print(D)
    D = np.diag(np.sum(D, axis=1))
    # D = np.diag(np.sum(W, axis=1))
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
    # print(sym_norm_Adj_matrix)
    return sym_norm_Adj_matrix # D^-0.5AD^-0.5

###test
# [[1. ,2. ,0. ,0.], [0. ,1., 5. ,0.], [0., 0. 1., 1.], [2., 0. ,0. ,1.]])
# adj=torch.Tensor([[0.,0.3,0.],[0.6,0.0,0.1],[0.3,0.2,0.0]])
# adj=sym_norm_Adj(adj)
# print(adj)
# adj=F.softmax(torch.from_numpy(adj),dim=1)
# print(adj)
#
# exit()

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, num_node,c_in,c_out,dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        global device
        self.in_channels=c_in
        self.dropout = nn.Dropout(p=dropout)
        # self.W_1 = torch.randn(c_in, requires_grad=True).to(device)
        # self.W_2 = torch.randn(num_node,num_node, requires_grad=True).to(device)
        # self.W_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        # self.b_s = torch.randn(1, num_node,num_node , requires_grad=True).to(device)
        # self.V_s = torch.randn(num_node,num_node, requires_grad=True).to(device)
        self.Wq=nn.Linear(c_in,c_in,bias=False)
        # nn.init.kaiming_uniform_(self.Wq.weight, nonlinearity="relu")
        self.Wk=nn.Linear(c_in,c_in,bias=False)
        # nn.init.kaiming_uniform_(self.Wk.weight, nonlinearity="relu")
        self.Wv=nn.Linear(c_in,num_node,bias=False)
        # nn.init.kaiming_uniform_(self.Wv.weight, nonlinearity="relu")
    def forward(self, x,score_his=None):
        '''
        :param x: (batch_size, N, C)
        :return: (batch_size, N, C)
        '''
        batch_size, num_of_vertices, in_channels = x.shape

        # Q K V 改之后
        Q=self.Wq(x)
        # print("Q:",Q.shape)
        K=self.Wk(x)
        # print("K:", K.shape)
        V=self.Wv(x)
        score = torch.matmul(Q, K.transpose(1, 2))
        score=F.softmax(score,dim=1)
        score=torch.matmul(score,V)
        # score=F.relu(score)
        # # print("V:", V.shape)
        # if score_his!=None:
        #     score = torch.matmul(Q, K.transpose(1, 2))+score_his  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        # else:
        #     score = torch.matmul(Q, K.transpose(1, 2))
        # score=torch.softmax(score,dim=-1)
        # score=torch.matmul(score,V)


        # 改之前
        # if score_his!=None:
        #     score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)#+score_his  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        #     # score_his = score
        # else:
        #     score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)
        # #
        # # score=F.softmax(score,dim=-1)
        # score=torch.sigmoid(score+self.b_s) # b n n + 1 n n = b n n
        # # # score=torch.softmax(score+self.b_s,dim=-1) # b n n + 1 n n = b n n
        # # # score=torch.softmax(score,dim=1)
        # # # score=torch.einsum("nn,bnn->bnn",self.V_s,score)
        # score=torch.matmul(self.V_s,score)
        # score=torch.softmax(score,dim=1)



        #normalization
        # score=score-torch.max(score,1,keepdim=True)[0]
        # exp=torch.exp(score)
        # score_norm=exp/torch.sum(exp,1,keepdim=True)
        # score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b, N, N)

        # print(score_norm)
        # 公式6  返回注意力和更新的score_his用于下一次传参
        # return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices)),score_his # (b t n n)
        return score,score_his # (b n n)



class spatialAttentionGCN(nn.Module):
    def __init__(self, Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        global device
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(Adj_matrix)).to(torch.float32)  # (N, N)
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix,dim=1)
        # self.W_s=torch.randn(1,requires_grad=True).to(device)
        # self.b_s=torch.randn(170,)
        # print(in_channels)
        # print(out_channels)
        self.static=nn.Linear(in_channels,out_channels,bias=True)
        # nn.init.kaiming_uniform_(self.static.weight, nonlinearity="relu")
        self.alpha = nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)  # D
        self.beta = nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)  # S
        # self.gamma = nn.Parameter(torch.FloatTensor([0.05]), requires_grad=True)
        # self.self_link = nn.Parameter(torch.FloatTensor([0.4]), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)  # 正向
        # self.beta2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)  # 转置

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta1 = nn.Linear(in_channels, in_channels, bias=True)
        # nn.init.kaiming_uniform_(self.Theta1.weight, nonlinearity="relu")
        # self.Theta2 = nn.Linear(in_channels, in_channels, bias=True)
        self.SAt = Spatial_Attention_layer(num_node=self.sym_norm_Adj_matrix.shape[0],c_in=in_channels,c_out=out_channels,dropout=dropout)
        # self.SAt_T = Spatial_Attention_layer(num_node=self.sym_norm_Adj_matrix.shape[0],c_in=in_channels,c_out=out_channels,dropout=dropout)
        # self.norm=nn.LayerNorm((64,self.sym_norm_Adj_matrix.shape[0],in_channels))

    def forward(self, x,score_his=None):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, C_in)
        :return: (batch_size, N, C_out)
        '''

        # batch_size, num_of_vertices, in_channels = x.shape
        global device
        spatial_attention,score_his = self.SAt(x,score_his)  # (batch, N, N) 分数st
        # spatial_attention_T,_ = self.SAt_T(x,score_his)  # (batch, N, N) 分数st
        # spatial_attention_T=spatial_attention_T.permute(0,2,1)
        #
        # # x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        # spatial_attention = spatial_attention.to(device)
        x = x.to(device)
        # print("x:",x.shape)
        # static_out=self.static(x)
        # print("st",static_out.shape)
        sym_norm_Adj_matrix=self.sym_norm_Adj_matrix.to(device)
        # sym_norm_Adj_matrix_T=sym_norm_Adj_matrix.permute(1,0)
        static_out=torch.einsum("nn,bnc->bnc",sym_norm_Adj_matrix,x)
        # static_out=F.relu(static_out)
        # print(static_out.shape)

        # dy_adj=torch.einsum("nn,bnn->bnn",sym_norm_Adj_matrix,spatial_attention)
        # dy_adj=torch.matmul(spatial_attention,sym_norm_Adj_matrix)
        dy_adj=torch.matmul(sym_norm_Adj_matrix,spatial_attention)
        dy_out=torch.einsum("bnn,bnc->bnc",dy_adj,x)

        # dy_adj_T=torch.matmul(sym_norm_Adj_matrix_T,spatial_attention_T)
        # dy_out_T=torch.einsum("bnn,bnc->bnc",dy_adj_T,x)
        # dy_out=self.alpha2*dy_out+self.beta2*dy_out_T

        dy_out=self.Theta1(dy_out)
        # dy_out=F.relu(dy_out)
        # dy_out2=torch.sigmoid(self.Theta2(dy_out))
        # dy_out=dy_out1*dy_out2
        # dy_out=torch.einsum("bnn,bnc->bnc",spatial_attention,x)
        # print("st:",static_out.shape)
        # print("dy:",dy_out.shape)
        st_dy_out=self.alpha*static_out+self.beta*dy_out#+self.gamma*x
        # st_dy_out=self.Theta2(st_dy_out)
        # st_dy_out=self.norm(st_dy_out)+x
        # st_dy_out=static_out
        # 公式7
        # return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x))),score_his
        # gcn_out=torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x) # n n,b n c_in->b n c_in
        # print("gcn_out:",gcn_out.shape)
        # gcn_out_linear=self.Theta(gcn_out) # (b, n, c_in)->(b, n, c_out)
        return F.relu(st_dy_out),score_his

    def sym_norm_Adj(self,W):
        '''
        compute Symmetric normalized Adj matrix

        Parameters
        ----------
        W: np.ndarray, shape is (N, N), N is the num of vertices

        Returns
        ----------
        Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
        '''
        W = W.to(device=torch.device('cpu'))
        assert W.shape[0] == W.shape[1]
        W = W.cpu().detach().numpy()

        N = W.shape[0]
        D = np.zeros([N, N], dtype=type(W[0][0]))

        W = W +  0.4* np.identity(N)  # 为邻居矩阵加上自连接
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                if W[i][j] != 0.:
                    D[i][j] = 1
        # print(D)
        D = np.diag(np.sum(D, axis=1))
        # D = np.diag(np.sum(W, axis=1))
        # print("D:",D)
        sym_norm_Adj_matrix = np.dot(np.sqrt(D), W)
        # print("*****")
        # print(sym_norm_Adj_matrix.device)
        # print(D.device)
        sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))
        # N = W.shape[0]
        # W = W + torch.from_numpy(np.identity(N)) # 为邻居矩阵加上自连接
        # # D = np.diag(np.sum(W,axis=1))
        # D = torch.diag(torch.sum(W,dim=1))
        # sym_norm_Adj_matrix = torch.dot(np.sqrt(D),W)
        # sym_norm_Adj_matrix = torch.dot(sym_norm_Adj_matrix,np.sqrt(D))
        # print(sym_norm_Adj_matrix)
        return sym_norm_Adj_matrix  # D^-0.5AD^-0.5

class emb_GCN(nn.Module):
    def __init__(self, Adj_matrix, in_channels, out_channels, dropout=.0):
        super(emb_GCN, self).__init__()
        global device
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(Adj_matrix)).to(torch.float32)  # (N, N)
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix,dim=1).to(device)
        # self.sym_norm_Adj_matrix_D = torch.from_numpy(sym_norm_Adj(Adj_matrix)[0]).to(torch.float32)  # (N, N)
        # self.sym_norm_Adj_matrix_D = F.softmax(self.sym_norm_Adj_matrix_D, dim=1).to(device)
        self.static=nn.Linear(in_channels,out_channels,bias=True)
        nn.init.kaiming_uniform_(self.static.weight,nonlinearity="relu")
        # nn.init.kaiming_uniform_()
        self.alpha = nn.Parameter(torch.FloatTensor([0.4]), requires_grad=True)  # D
        self.beta = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)  # S
        self.gamma = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)

        # self.alpha2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)  # 正向
        # self.beta2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)  # 转置

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, in_channels, bias=False)
        nn.init.kaiming_uniform_(self.Theta.weight, nonlinearity="relu")
        # self.SAt = Spatial_Attention_layer(num_node=self.sym_norm_Adj_matrix.shape[0],c_in=in_channels,c_out=out_channels,dropout=dropout)
        # self.SAt_T = Spatial_Attention_layer(num_node=self.sym_norm_Adj_matrix.shape[0],c_in=in_channels,c_out=out_channels,dropout=dropout)
        # self.norm=nn.LayerNorm((64,self.sym_norm_Adj_matrix.shape[0],in_channels))
        # self.ln_res=SublayerConnection(64,0.2,True,True)
        self.E1=torch.randn(Adj_matrix.shape[0],Adj_matrix.shape[0],requires_grad=True).to(device)
        self.E2=torch.randn(Adj_matrix.shape[0],Adj_matrix.shape[0],requires_grad=True).to(device)
    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, C_in)
        :return: (batch_size, N, C_out)
        '''


        # batch_size, num_of_vertices, in_channels = x.shape
        global device
        x = x.to(device)
        # N=self.sym_norm_Adj_matrix.shape[0]
        adj = torch.matmul(self.E1, self.sym_norm_Adj_matrix)  # E1A
        adj = torch.matmul(adj, self.E2)  # E1AE2
        adj = F.softmax(adj,dim=1)


        static_out=torch.einsum("nn,bnc->bnc",self.sym_norm_Adj_matrix,x)

        dy_out=torch.einsum("nn,bnc->bnc",adj,x)


        dy_out=self.Theta(dy_out)
        # st_dy_out=self.alpha*static_out+self.beta*dy_out+self.gamma*x
        # return F.relu(st_dy_out)
        return F.relu(dy_out)

    def sym_norm_Adj(self,W):
        '''
        compute Symmetric normalized Adj matrix

        Parameters
        ----------
        W: np.ndarray, shape is (N, N), N is the num of vertices

        Returns
        ----------
        Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
        '''
        W = W.to(device=torch.device('cpu'))
        assert W.shape[0] == W.shape[1]
        W = W.cpu().detach().numpy()

        N = W.shape[0]
        D = np.zeros([N, N], dtype=type(W[0][0]))

        W = W +  0.4* np.identity(N)  # 为邻居矩阵加上自连接
        for i in range(W.shape[0]):
            for j in range(W.shape[0]):
                if W[i][j] != 0.:
                    D[i][j] = 1
        # print(D)
        D = np.diag(np.sum(D, axis=1))
        # D = np.diag(np.sum(W, axis=1))
        # print("D:",D)
        sym_norm_Adj_matrix = np.dot(np.sqrt(D), W)
        # print("*****")
        # print(sym_norm_Adj_matrix.device)
        # print(D.device)
        sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))
        # N = W.shape[0]
        # W = W + torch.from_numpy(np.identity(N)) # 为邻居矩阵加上自连接
        # # D = np.diag(np.sum(W,axis=1))
        # D = torch.diag(torch.sum(W,dim=1))
        # sym_norm_Adj_matrix = torch.dot(np.sqrt(D),W)
        # sym_norm_Adj_matrix = torch.dot(sym_norm_Adj_matrix,np.sqrt(D))
        # print(sym_norm_Adj_matrix)
        return sym_norm_Adj_matrix # D^-0.5AD^-0.5

class AVWGCN2(nn.Module):
    def __init__(self, dim_in, dim_out, Adj):
        super(AVWGCN2, self).__init__()
        # self.W = nn.Linear(dim_in, dim_out,bias=True).to(device=torch.device('cuda'))  # y = W * x
        # self.b = nn.Parameter(torch.Tensor(dim_out)).to(device=torch.device('cuda'))
        # self.W = nn.Linear(dim_in, dim_out,bias=True)  # y = W * x
        # self.b = nn.Parameter(torch.Tensor(dim_out))

        # torch.nn.init.normal_(self.W.weight, mean=0, std=1)
        # torch.nn.init.normal_(self.b, mean=0, std=1)
        self.adj=Adj
        self.sp_att_gcn=spatialAttentionGCN(self.adj,dim_in,dim_out)
        # self.emb_gcn=emb_GCN(self.adj,dim_in,dim_out)
        self.linear=nn.Linear(dim_in,dim_out,bias=True)
        # nn.init.kaiming_uniform_(self.linear.weight,nonlinearity='relu')
        self.dropout=nn.Dropout(0.1)
        # self.att_his=None
    def forward(self, x, node_embeddings=0):
        # 静态
        # print(x.shape)
        # print("&&&&&&&&&&&&&&&&&")
        # N=x.shape[1]
        # h=self.W(x) #[b n c_in] --> [b n c_out]
        # static_out=torch.einsum('nn,bnc->bnc',self.adj,x)#+self.b
        # static_out=self.linear(static_out) # [b n c_in] ==> [b n c_out]
        # print(static_out.shape)
        # static_out=F.softmax(static_out,dim=2)

        gcn_out,att_his=self.sp_att_gcn(x)
        # gcn_out2=self.emb_gcn(x)
        # global att_his
        # gcn_out,att_his=self.sp_att_gcn(x,self.att_his)
        # self.att_his=score_his
        # emb_out=self.emb_net(x,node_embeddings)
        # print("*********")
        # print(gcn_out.shape,emb_out.shape)
        # gcn_out=self.alpha*gcn_out+self.beta*emb_out
        # static_out_32=torch.tensor(static_out,dtype=torch.float32)
        # static_out=torch.as_tensor(static_out, dtype=torch.float32)
        # print("dyout:",dy_out.dtype)
        # print("stout:",static_out.dtype)
        static_dy_out=self.linear(gcn_out)
        # static_dy_out=self.dropout(static_dy_out)

        return F.relu(static_dy_out)


if __name__=='__main__':
    x=torch.randn(32, 10, 1).to(torch.float32)
    graph=torch.randn(10,10).to(torch.float32)
    # print(graph)
    mode_emb=torch.ones(170,2)
    input_dim = 1
    output_dim = 1
    embed_dim = 2
    cheb_k=2
    # update = AVWGCN(input_dim+1, output_dim, cheb_k, embed_dim)
    # output=update(x,mode_emb)
    # print(output.shape)
    print("****")
    # sp_att_layer=Spatial_Attention_layer()
    # sp_out=sp_att_layer(x)
    # print(sp_out.shape) # [32, 278, 278]
    # sp_att_GCN=spatialAttentionGCN(graph,input_dim,output_dim)
    # sp_att_gcn_out=sp_att_GCN(x)
    # print(sp_att_gcn_out[0].shape)
    static_dy_gcn=AVWGCN2(input_dim,output_dim,graph)
    out=static_dy_gcn(x)
    print(out.shape)
    # print(out[1].shape)
