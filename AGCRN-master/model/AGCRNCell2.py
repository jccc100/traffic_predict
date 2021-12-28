import torch
import torch.nn as nn
from model.AGCN2 import AVWGCN,AVWGCN2
from model.AGCN2 import spatialAttentionGCN

device=torch.device('cuda')
# device=torch.device('cpu')

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, Adj,cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.adj=Adj
        # self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        # self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.gate=AVWGCN2(dim_in+self.hidden_dim,2*dim_out,self.adj)
        self.update=AVWGCN2(dim_in+self.hidden_dim,dim_out,self.adj)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h



    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class AGCRNCell2(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, Adj):
        super(AGCRNCell2, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.adj=Adj
        # self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        # self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)
        # self.gate_z=AVWGCN2(dim_in,dim_out,self.adj)
        # self.gate_r=AVWGCN2(dim_in,dim_out,self.adj)
        self.gate = AVWGCN2(dim_in + self.hidden_dim, 2 * dim_out, self.adj)
        self.update=AVWGCN2(dim_in+self.hidden_dim,dim_out,self.adj)
        # self.gate_cnn1=nn.Conv1d(dim_in,self.hidden_dim,kernel_size=3,stride=1,padding=2,dilation=2,bias=True)
        # self.gate_cnn2=nn.Conv1d(dim_in,self.hidden_dim,kernel_size=3,stride=1,padding=2,dilation=2,bias=True)
        # self.alpha = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)  # D
        # self.beta = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)  # S
        # self.GAT = spatialAttentionGCN(self.adj, dim_in, self.hidden_dim, dropout=.0)
    def forward(self, x, state, node_embeddings):
        global device
        # print("cell:",x.shape)
        # print("cell_state:",state.shape)
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        # state = state.to(x.device)
        # print("state:",state.shape)
        state=state.to(device)
        x=x.to(device)
        # GAT_input=x
        # B,N,C=x.shape
        # gate_cnn_input=x.reshape(B,C,N)
        # gate_cnn_input=x.permute(0,2,1)
        # gate_cnn_out=torch.tanh(self.gate_cnn1(gate_cnn_input))*torch.sigmoid(self.gate_cnn2(gate_cnn_input))
        # gate_cnn_out=gate_cnn_out.permute(0,2,1)
        # print("gate_out:",gate_cnn_out.shape)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        # z=torch.sigmoid(self.gate_z(input_and_state,node_embeddings))
        # r=torch.sigmoid(self.gate_r(input_and_state,node_embeddings))
        z=z.to(device)
        r=r.to(device)
        
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        # GAT_out=self.GAT(GAT_input)[0]+GAT_input
        # print("aa:",h.shape)
        # print("bb:",GAT_out[0].shape)
        # h=h+GAT_out
        # print("cell_h:",h.shape)
        # h=self.alpha*h+self.beta*gate_cnn_out
        return h



    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

if __name__=='__main__':
    x = torch.randn(32, 170, 1).to(torch.float32)
    state=torch.ones(size=x.shape)
    graph = torch.ones(170, 170).to(torch.float32)
    # print(graph)
    node_emb = torch.ones(170, 2)
    input_dim = 1
    output_dim = 1
    embed_dim = 2
    cheb_k = 2
    agcrncell=AGCRNCell(x.shape[1],dim_in=input_dim,dim_out=output_dim,Adj=graph,cheb_k=cheb_k,embed_dim=embed_dim)
    out1=agcrncell(x,state,node_emb)
    print(out1.shape)
    print("***")
    cell=AGCRNCell2(x.shape[1],input_dim,output_dim,graph)
    out2=cell(x,state,node_emb)
    print(out2.shape)
