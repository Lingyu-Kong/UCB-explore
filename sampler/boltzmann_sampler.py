import numpy as np
import torch
from torch.nn import Softmax
import torch.nn.functional as F
import time
import wandb
import matplotlib.pyplot as plt

from sampler.uncertainty_critic import Count_Based_Uncertainty,RND_Uncertainty
from sampler.value_critic import ValueCritic
from sampler.memory import Memory,MaxMemory
from sampler.env import Env
from sampler.random_generator import Random_Generator
from utils.tensor_utils import to_tensor

uncertainty_type={
    "count_based":Count_Based_Uncertainty,
    "rnd":RND_Uncertainty,
}

class BoltzmannSampler(object):
    def __init__(
        self,
        uncertainty:str,
        value_critic_params:dict,
        uncertainty_critic_params:dict,
        memory_params:dict,
        env_params:dict,
        generator_params:dict,
        num_atoms:int,
        alpha:float,
        use_dataset:bool,
        dataset_path:str,
        if_memory_update:bool,
        if_greedy:bool,
    ):
        self.memory = MaxMemory(**memory_params)
        self.value_critic = ValueCritic(**value_critic_params)
        self.uncertainty_critic = uncertainty_type[uncertainty](**uncertainty_critic_params)
        self.env=Env(**env_params)
        self.generator=Random_Generator(**generator_params)
        self.num_atoms=num_atoms
        self.alpha=alpha
        self.ucbs=np.zeros(self.memory.memory_size)
        self.use_dataset=use_dataset
        self.dataset_path=dataset_path
        self.if_memory_update=if_memory_update
        self.if_greedy=if_greedy

    def memory_init(self):
        if not self.use_dataset:
            states=np.zeros((self.memory.memory_size,self.num_atoms,3))
            for i in range(self.memory.memory_size):
                retry=True
                while retry:
                    retry=False
                    state=self.generator.conform_sample()
                    for j in range(i):
                        distance=np.linalg.norm((states[j].reshape(-1))-(state.reshape(-1)),ord=2,axis=0)
                        if distance<5:
                            retry=True
                            break
                    if not retry:
                        states[i]=state
                print("found state:",i)
            self.memory.init_state_memory(states)
        else:
            states=np.loadtxt(open(self.dataset_path),delimiter=",",skiprows=0)
            states=np.reshape(states,(-1,self.num_atoms,3))
            choices=np.random.choice(states.shape[0],size=self.memory.memory_size,replace=False)
            self.memory.state_memory=states[choices]
    
    def ucb_init(self,batch_size):
        left=0
        while left<self.memory.memory_size:
            interval=np.arange(left,min(self.memory.memory_size,left+batch_size))
            states=self.memory.state_memory[interval]
            if self.if_memory_update:
                v_values=-self.value_critic(to_tensor(states)).view(-1)
            else:
                v_values=-to_tensor(self.memory.compute_average_value(interval))
            if self.if_greedy:
                u_values=self.uncertainty_critic.forward(to_tensor(states),interval).view(-1)
                self.ucbs[interval]=(v_values+self.alpha*u_values).detach().cpu().numpy()
            else:
                self.ucbs[interval]=v_values.detach().cpu().numpy()
            left+=batch_size

    def sample(self):
        probs=Softmax(dim=0)(10*to_tensor(self.ucbs)).detach().cpu().numpy()
        indexs=list(range(self.memory.memory_size))
        index=np.random.choice(indexs,p=probs)
        # index=np.argmax(self.ucbs)
        # index=np.random.randint(0,self.memory.memory_size)
        state=self.memory.read_state(index)
        return state,index

    def ucb_update(self,batch_size):
        choices=np.random.choice(self.memory.memory_size,size=batch_size,replace=False)
        states=self.memory.state_memory[choices]
        if self.if_memory_update:
            v_values=-self.value_critic(to_tensor(states)).view(-1)
        else:
            v_values=-to_tensor(self.memory.compute_average_value(choices))
        if self.if_greedy:
            u_values=self.uncertainty_critic.forward(to_tensor(states),choices).view(-1)
            self.ucbs[choices]=(v_values+self.alpha*u_values).detach().cpu().numpy()
        else:
            self.ucbs[choices]=v_values.detach().cpu().numpy()

    def memory_update(self,state):
        index=self.memory.store_state(state)
        self.uncertainty_critic.update(state,index)

    def find_min_energy(self):
        states=self.memory.state_memory
        min_energy=0
        for i in range(self.memory.memory_size):
            energy=self.env.compute(states[i].tolist())
            self.memory.store_reward(i,energy)
            if energy<min_energy:
                min_energy=energy
        return min_energy
    
    def train(
        self,
        num_steps:int,
        update_batch_size:int,
    ):
        self.memory_init()
        self.ucb_init(update_batch_size)
        min_energy=self.find_min_energy()
        for i in range(num_steps):
            start_time=time.time()
            state,index=self.sample()
            epsilon=self.generator.action_sample(state)
            energy=self.env.compute((state+epsilon).tolist())
            self.memory.store_reward(index=index,reward=energy)
            rewards=self.memory.read_rewards(index=index)
            if self.if_memory_update:
                v_value=self.value_critic(to_tensor(state).unsqueeze(0))
                v_value=torch.repeat_interleave(v_value,rewards.shape[0],dim=0).squeeze(-1)
                v_loss=F.mse_loss(v_value,to_tensor(rewards))
                self.value_critic.optimizer.zero_grad()
                v_loss.backward()
                self.value_critic.optimizer.step()
                self.value_critic.scheduler.step()
            self.uncertainty_critic.step(state,index)
            # if self.if_memory_update:
            #     self.memory_update(state_prime)
            self.ucb_update(update_batch_size)
            if energy<min_energy:
                min_energy=energy
            end_time=time.time()
            wandb.log({"index":index,
                       "min_energy":min_energy,})
            if self.if_memory_update:
                wandb.log({"v_loss":v_loss.item(),})

            if i%10000==9999:
                choices=np.random.choice(self.memory.memory_size,size=update_batch_size,replace=False)
                states=self.memory.state_memory[choices]
                v_values=(-self.memory.compute_average_value(choices)).tolist()
                u_values=(self.alpha*self.uncertainty_critic.forward(to_tensor(states),choices).view(-1)).tolist()
                plt.figure()
                plt.plot(v_values,label="v_values")
                plt.plot(u_values,label="u_values")
                plt.legend()
                wandb.log({"steps_"+str(i):plt})

                print("Steps: {}/{}, Time: {}, Energy Found: {}".format(i,num_steps,end_time-start_time,energy))
        
        return min_energy