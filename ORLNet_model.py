import math
import sys

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

class CombineNet(nn.Module):
    def __init__(self, n, nfeats, n_view, n_clusters, block, para1, para2, alpha, lamb, device, fusion_type):
        super(CombineNet, self).__init__()
        self.n_clusters = n_clusters
        self.block = block
        self.n_view = n_view
        self.nfeats = nfeats
        self.n = n
        self.lamb = torch.FloatTensor([lamb]).to(device)
        self.alpha = torch.FloatTensor([alpha]).to(device)
        self.ZZ_init = []
        for ij in range(n_view):
            self.ZZ_init.append(nn.Linear(nfeats[ij], n_clusters).to(device))
        self.theta1 = nn.Parameter(torch.FloatTensor([para1]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([para2]), requires_grad=True).to(device)
        self.U = nn.Linear(n_clusters, n_clusters).to(device)
        self.W= nn.Linear(n_clusters, n_clusters).to(device)
        self.device = device
        self.S_views=nn.ModuleList([nn.Linear(nfeats[i], nfeats[i]).to(device) for i in range(n_view)])
        self.fusionlayer = FusionLayer(n_view, fusion_type, n_clusters, hidden_size=64)


    def soft_threshold(self, u, theta):
        return F.selu(u - theta) - F.selu(-1.0 * u - theta)

    def forward(self, features, lap_Z, lap_G):
        Z = list()
        out_tmp = 0
        for j in range( self.n_view):
            out_tmp += self.ZZ_init[j](features[j] / 1.0)
        Z.append(out_tmp / self.n_view)
        H_list = list()
        LG = torch.norm(Z[-1].t().matmul(Z[-1]))
        Gs={}
        for i in range(self.n_view):
            Gs[i]=list()
            Gs[i].append(self.soft_threshold(torch.mm(Z[-1].t(), features[i])/LG.to(self.device), self.theta2))
        for i in range(self.block):
            H_list=list()
            for j in range(self.n_view):
                input_Z1 = self.U(Z[-1])
                LZ = torch.norm(Gs[j][-1].t().matmul(Gs[j][-1]))
                input_Z2=torch.mm(lap_Z[j],Z[-1])*self.alpha/LZ
                input_Z3=self.W(torch.mm(features[j], Gs[j][-1].t()))
                H_list.append(input_Z1 -  input_Z2  +  input_Z3)

                LG = torch.norm(Z[-1].t().matmul(Z[-1]))
                input_G1=self.S_views[j](Gs[j][-1])
                input_G2=torch.mm(Gs[j][-1], lap_G[j])*self.lamb/LG
                input_G3=torch.mm(Z[-1].t(), features[j])/LG
                Gs[j].append(self.soft_threshold(input_G1 - input_G2 + input_G3, self.theta2))
            z=self.fusionlayer(H_list)
            Z.append(self.soft_threshold(z, self.theta1))
        return Z[-1],H_list

