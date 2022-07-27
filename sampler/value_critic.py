import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from gnn.ignn import IGNN
from gnn.metalayer import MLPwoLastAct

class ValueCritic(nn.Module):
    def __init__(
        self,
        gnn_params:dict,
        mlp_params:dict,
        lr:float,
        decay_interval:int,
        decay_rate:float,
        device:torch.device,
    ):
        super().__init__()
        self.gnn = IGNN(**gnn_params)
        self.final_global_mlp = MLPwoLastAct(**mlp_params)
        self.device=device
        self.gnn.to(self.device)
        self.final_global_mlp.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, decay_interval, decay_rate)
        self.weight_init(self.final_global_mlp)
        self.weight_init(self.gnn)

    def forward(self,states):
        _,_,global_attr=self.gnn(states)
        value=self.final_global_mlp(global_attr)
        return value  ## [batch_size,1]

    def save_model(self,path):
        torch.save(self.state_dict(),path)
    
    def load_model(self,path):
        self.load_state_dict(torch.load(path))

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)