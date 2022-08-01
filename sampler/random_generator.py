import numpy as np

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edge_index = [rows, cols]
    return edge_index

class Random_Generator(object):
    def __init__(
        self,
        num_atoms:int,
        pos_scale:float,
        action_scale:float,
        threshold:float,
        action_policy:str,
    ):
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.action_scale=action_scale
        self.threshold=threshold
        self.action_policy=action_policy
        self.edge_index=get_edges(num_atoms)

    def conform_sample(self):
        pos=np.zeros((self.num_atoms,3))
        for i in range(self.num_atoms):
            if_continue=True
            while if_continue:
                new_pos=np.random.rand(3)*2*self.pos_scale-self.pos_scale
                if_continue=False
                for j in range(i):
                    distance=np.linalg.norm(new_pos-pos[j],ord=2)
                    if distance<self.threshold:
                        if_continue=True
                        break
            pos[i,:]=new_pos
        return pos

    def unconstrained_action_sample(self):
        if self.action_policy=='uniform':
            return np.random.rand(self.num_atoms,3)*2*self.action_scale-self.action_scale
        elif self.action_policy=='gaussian':
            return np.random.randn(self.num_atoms,3)*2*self.action_scale-self.action_scale

    def action_sample(self,state):
        while True:
            action=self.unconstrained_action_sample()
            state_prime=state+action
            rows=state_prime[self.edge_index[0]]
            cols=state_prime[self.edge_index[1]]
            distance=np.linalg.norm(rows-cols,ord=2,axis=1)
            if np.all(distance>=self.threshold):
                return action