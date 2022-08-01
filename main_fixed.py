import numpy as np
import wandb

import config_fixed as config
from sampler.boltzmann_sampler import BoltzmannSampler
from sampler.maxucb_sampler import MaxUCBSampler

wandb.login()
wandb.init(project="go-explore", config=config.shared_params)

Sampler={
    'boltzmann':BoltzmannSampler,
    'maxucb':MaxUCBSampler,
}

shared_params=config.shared_params
sampler_type=shared_params["sampler_policy"]
if sampler_type=="boltzmann":
    sampler_params=config.boltzmann_params
elif sampler_type=="maxucb":
    sampler_params=config.maxucb_params

training_params=config.training_params
path_file=config.path_file

def log_history_path(path_file):
    history_path=np.loadtxt(open(path_file),delimiter=",",skiprows=0)
    for i in range(history_path.shape[0]):
        wandb.log({"min_energy":history_path[i]})

if __name__=="__main__":
    sampler=Sampler[sampler_type](**sampler_params)
    # log_history_path(path_file)
    sampler.train(**training_params)
    sampler.value_critic.save_model("./model_save/value_critic.pt")
    state,_=sampler.sample()
    print("==========================================================")
    print("final decision: ",state)