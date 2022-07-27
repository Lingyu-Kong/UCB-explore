import torch
import torch.nn as nn
import numpy as np

from utils.tensor_utils import to_tensor
from gnn.ignn import IGNN

class Count_Based_Uncertainty(object):
    def __init__(
        self,
        state_num:int,
    ):
        self.counts=np.zeros(state_num)
        self.total_count=1
    
    def forward(self,states,interval):
        counts=self.counts[interval]
        u_values=np.sqrt(2*np.log(self.total_count)/(counts+1e-4))
        return to_tensor(u_values)

    def step(self,state,index):
        self.counts[index]+=1
        self.total_count+=1

    def update(self,state,index):
        self.count[index]=0

class RND_Uncertainty(nn.Module):
    def __init__(
        self,
        gnn_params:dict,
        lr:float,
        update_per_step:int,
        device:torch.device,
    ):
        super().__init__()
        self.gnn=IGNN(**gnn_params)
        self.gnn_target=IGNN(**gnn_params)
        self.device=device
        self.gnn.to(device)
        self.gnn_target.to(device)
        self.optimizer=torch.optim.Adam(self.gnn.parameters(),lr=lr)
        self.update_per_step=update_per_step
        self.weight_init(self.gnn)
        self.weight_init(self.gnn_target)

    def forward(self,states,interval):
        _,_,global_attr=self.gnn(states)
        _,_,global_attr_target=self.gnn_target(states)
        u_values=torch.norm(global_attr-global_attr_target,dim=1)
        return u_values

    def step(self,state,index):
        for _ in range(self.update_per_step):
            _,_,global_attr=self.gnn(to_tensor(state).unsqueeze(0))
            global_attr=global_attr.squeeze(0)
            _,_,global_attr_target=self.gnn_target(to_tensor(state).unsqueeze(0))
            global_attr_target=global_attr_target.squeeze(0)
            u_loss=torch.norm(global_attr-global_attr_target,dim=0)
            self.optimizer.zero_grad()
            u_loss.backward()
            self.optimizer.step()

    def update(self,state,index):
        pass

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
