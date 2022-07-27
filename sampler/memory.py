import numpy as np

class Memory(object):
    def __init__(
        self,
        memory_size:int,
        num_atoms:int,
        time_to_live:int,
    ):
        self.state_memory = np.zeros((memory_size, num_atoms, 3))
        self.reward_memory = np.zeros((memory_size, time_to_live))
        self.reward_memory_top = np.zeros(memory_size,dtype=int)
        self.memory_top=0
        self.time_to_live = time_to_live
        self.memory_size = memory_size

    def init_state_memory(self,states):
        assert states.shape[0] == self.state_memory.shape[0]
        for i in range(states.shape[0]):
            self.state_memory[i] = states[i]

    def store_state(self,state):
        self.state_memory[self.memory_top] = state
        self.reward_memory[self.memory_top]=np.zeros(self.time_to_live)
        self.reward_memory_top[self.memory_top]=0
        tmp=self.memory_top
        self.memory_top = (self.memory_top + 1) % self.memory_size
        return tmp

    def store_reward(self,index,reward):
        self.reward_memory[index][self.reward_memory_top[index]%self.time_to_live] = reward
        self.reward_memory_top[index] += 1

    def read_state(self,index):
        state=self.state_memory[index]
        return state
    def read_rewards(self,index):
        rewards=self.reward_memory[index][0:min(self.reward_memory_top[index],self.time_to_live)]
        return rewards

    def compute_average_value(self,indexs):
        rewards=self.reward_memory[indexs]
        tops=np.clip(self.reward_memory_top[indexs],1e-4,self.time_to_live-1)
        return np.sum(rewards,axis=1)/tops
    