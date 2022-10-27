import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import numpy as np

class GraphReccurent(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.seq = nn.LSTM(input_size=in_features,
                          num_layers=1,
                          hidden_size=out_features,
                          batch_first=False, bidirectional=False)

    def forward(self, x, adj):
        neighbor = torch.mm(adj, x)
        seq = torch.stack((neighbor, x), dim=0) # (2, bsz, in_features)
        output, (hn, cn) = self.seq(seq)        # (2, bsz, out_features)  (1, bsz, out_features)
        node = hn.squeeze(0)
        return node

class Interact(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        ## GRN
        self.grn1 = GraphReccurent(input_size, args.hidden_size)
        self.grn2 = GraphReccurent(input_size, args.hidden_size)  
        self.node_size = args.hidden_size * 2
        ## pooling

    @staticmethod
    def _normalize(A, symmetric=True):
        #I = torch.eye(A.size(0)).to(A.device)
        #A = A + I
        d = A.sum(1)
        if symmetric:            
            D = torch.diag(torch.pow(d, -0.5)) #D = D^-1/2
            A_hat = D.mm(A).mm(D)
        else :            
            D =torch.diag(torch.pow(d, -1)) # D=D^-1
            A_hat = D.mm(A)
            #A_hat = A_hat.mm(A_hat) + I
        return A_hat

    @staticmethod
    def _construct_graph(structure):
        mids, cids, pids, time = structure
        # split according to pid=='None'
        source_index = [i for i in range(len(pids)) if pids[i] == 'None']
        temp = source_index + [len(pids)]
        cas_index = [list(range(temp[i], temp[i+1])) for i in range(len(source_index))]
        mes_num = len(mids)
        A1 = torch.zeros((mes_num, mes_num)).long()
        A2 = torch.zeros((mes_num, mes_num)).long()
        for cas_i in range(len(cas_index)):
            indexes = cas_index[cas_i]
            mids_array = np.array(mids)
            pids_array = np.array(pids)
            mid_index_dict = dict(zip(mids_array[indexes], indexes))
            for i in indexes:
                j = mid_index_dict.get(pids[i], i) # if pid=='None' get i => add self-loop
                A1[i, j] = 1  # parent
                A2[j, i] = 1 if i != j else 0  # children
        A2_sum = A2.sum(dim=1)
        for i in range(len(A2_sum)):
            if A2_sum[i] == 0:
                A2[i, i] = 1  # add self-loop to leaves
        return A1, A2
    
    def forward(self, mes_embed, structure):
        A1, A2 = self._construct_graph(structure)
        A1_hat = self._normalize(A1.float().to(mes_embed.device), symmetric=False)
        A2_hat = self._normalize(A2.float().to(mes_embed.device), symmetric=False)
        mes_p = self.grn1(mes_embed, A1_hat) # (bsz, hidden)
        mes_c = self.grn2(mes_embed, A2_hat) # (bsz, hidden)
        mes_update = torch.cat((mes_p, mes_c), dim=-1) # (bsz, hidden*2)
        return mes_update
