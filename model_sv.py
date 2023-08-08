import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import SAGEConv
from copy import deepcopy

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=10, out_channels=2, embedding_size=128, k=5):
        super(GNN, self).__init__()
        self.k = k

        self.hidden_channels=hidden_channels
        self.out_channels=out_channels
        self.SVNet = SVNet(hidden_channels, out_channels, num_layers)
        
        self.embedding = nn.Embedding(in_channels, hidden_channels)

        self.mlp = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            BatchNorm1d(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )

        self.mlp_inloop = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels),
        )

        self.state_emb = nn.Embedding(5, hidden_channels)
        # 0 atom false
        # 1 atom true
        # 2 atom unknown
        # 3 inloop false
        # 4 inloop true
    
    def forward(self, x, edge_index, trace, batch, u_index, inloop, predict_time, atom_mask):

        # x (node_size, 8)
        # edge_index (2, edge_size)
        # trace (node_size, 5)
        # u_index (graph_size)
        # inloop (node_size, 5)
        # predict_time (batch_size)
        # atom_mask (node_size)
        
        device=x.device
        h = self.embedding(x)

        atom = []
        state = []
        
        a = deepcopy(atom_mask).to(device)
        b =  torch.ones(atom_mask.shape[0]).to(device)
        c =  torch.zeros(atom_mask.shape[0]).to(device)
        b = b * 2
        a = torch.where(a!=0, a, b)
        un_atom = torch.where(a==1, c, a).long()

        inloop=inloop+3
        t = self.state_emb(un_atom)+self.state_emb(torch.ones(atom_mask.shape[0], dtype=torch.long, device=device)*3)
        for i in range(self.k):
            h, a, s = self.SVNet(self.mlp(torch.cat((h, t), dim=-1)), edge_index, atom_mask.unsqueeze(dim=1), u_index)
            t = self.state_emb(trace[:, i]+un_atom)+self.state_emb(inloop[:, i])
            atom.append(a)
            state.append(s)
        
        batch_size=predict_time.shape[0]
        predict_range=torch.arange(batch.shape[0], dtype=torch.long, device=device)
        batch_range=torch.arange(batch_size, dtype=torch.long, device=device)
        predict_trace = torch.stack(atom, dim=1)[predict_range, predict_time[batch], :]
        predict_inloop= self.mlp_inloop(torch.stack(state, dim=1).reshape(-1, self.hidden_channels)).reshape(-1, self.k, self.out_channels)[batch_range, predict_time, :]

        return predict_trace, predict_inloop
    
    
class SVNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(SVNet, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(SAGEConv(in_channels, in_channels))
            self.bn_layers.append(BatchNorm1d(in_channels))

        self.atom_readout = nn.Sequential(
            Linear(in_channels, in_channels),
            BatchNorm1d(in_channels),
            nn.ReLU(),
            Linear(in_channels, out_channels),
            # nn.Softmax(dim=-1),
        )
        
    def forward(self, x, edge_index, atom_mask, u_index):
        # print(trace.size())     # [B*num_node, max_trace_len]
        # print(loop.size())      # [B, max_trace_len]
        # print(data.loop.size())     # [B, max_trace_len]
        # print(data.trace.size())    # [B*num_node, max_trace_len]
        h = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)

        state = []
        for u in u_index:
            state.append(h[u.item()])
        state = torch.stack(state, dim=0)
        atom = self.atom_readout(h) * atom_mask

        return h, atom, state