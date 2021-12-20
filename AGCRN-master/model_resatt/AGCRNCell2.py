import torch
import torch.nn as nn
from model_resatt.AGCN2 import AVWGCN,AVWGCN2

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
        self.att_his = torch.zeros((64,170,170),requires_grad=False).to(device=device)
        # self.norm = nn.LayerNorm((64, self.sym_norm_Adj_matrix.shape[0], in_channels))
        # self.conv1D=nn.Conv1d(dim_in,dim_out,3)
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
        self.att_his.detach_()
        input_and_state = torch.cat((x, state), dim=-1)
        gate_res, self.att_his = self.gate(input_and_state, self.att_his)
        z_r = torch.sigmoid(gate_res)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        # z=torch.sigmoid(self.gate_z(input_and_state,node_embeddings))
        # r=torch.sigmoid(self.gate_r(input_and_state,node_embeddings))
        z=z.to(device)
        r=r.to(device)
        candidate = torch.cat((x, z*state), dim=-1)
        update_res, self.att_his = self.update(candidate, self.att_his)
        hc = torch.tanh(update_res)
        h = r*state + (1-r)*hc
        # print("cell_h:",h.shape)
        return h



    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

if __name__=='__main__':
    x = torch.randn(64, 170, 1).to(torch.float32)
    state=torch.ones(size=x.shape)
    graph = torch.ones(170, 170).to(torch.float32)
    # print(graph)
    node_emb = torch.ones(170, 2)
    input_dim = 1
    output_dim = 1
    embed_dim = 2
    cheb_k = 2
    # agcrncell=AGCRNCell(x.shape[1],dim_in=input_dim,dim_out=output_dim,Adj=graph,cheb_k=cheb_k,embed_dim=embed_dim)
    # out1=agcrncell(x,state,node_emb)
    # print(out1.shape)
    print("***")
    cell=AGCRNCell2(x.shape[1],input_dim,output_dim,graph)
    out2=cell(x,state,node_emb)
    print(out2.shape)
