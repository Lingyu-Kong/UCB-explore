import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampler_policy=["boltzmann","maxucb"]
uncertainty_policy=["count_based","rnd"]
epsilon_policy=["gaussian","uniform"]

shared_params={
    "device":device,
    "num_atoms":10,
    "pos_scale": 2.0,
    "action_scale": 1.0,
    "memory_size":5000,
    "sampler_policy":"boltzmann",
    "uncertainty_policy":"count_based",
    "epsilon_policy":"uniform",
    "use_dataset":False,
    "if_replace":False,
    "if_greedy":True,
}

data_file="./"+str(shared_params["num_atoms"])+".csv"
path_file="./"+str(shared_params["num_atoms"])+"_path.csv"

training_params={
    "num_steps":100000,
    "update_batch_size":5000,
}

gnn_params = {
    "device":device,
    "num_atoms":shared_params["num_atoms"],
    "mlp_hidden_size":256,
    "mlp_layers":2,
    "latent_size":64,
    "use_layer_norm":False,
    "num_message_passing_steps":6,
    "global_reducer":"sum",
    "node_reducer":"sum",
    "dropedge_rate":0.1,
    "dropnode_rate":0.1,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
    "cycle":1,
    "node_attn":True,
    "global_attn":True,
}

mlp_params = {
    "input_size":gnn_params["latent_size"],
    "output_sizes":[gnn_params["latent_size"]]*2+[1],
    "use_layer_norm":False,
    "activation":nn.ReLU,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
}

count_based_params = {
    "state_num":shared_params["memory_size"],
}

RND_params = {
    "gnn_params":{
        "device":device,
        "num_atoms":shared_params["num_atoms"],
        "mlp_hidden_size":256,
        "mlp_layers":2,
        "latent_size":64,
        "use_layer_norm":False,
        "num_message_passing_steps":2,
        "global_reducer":"sum",
        "node_reducer":"sum",
        "dropedge_rate":0.1,
        "dropnode_rate":0.1,
        "dropout":0.1,
        "layernorm_before":False,
        "use_bn":False,
        "cycle":1,
        "node_attn":True,
        "global_attn":True,
    },
    "lr":0.001,
    "update_per_step":5,
    "device":device,
}

uncertainty_critic_params={
    "count_based":count_based_params,
    "rnd":RND_params,
}

boltzmann_params={
    "uncertainty":shared_params["uncertainty_policy"],
    "value_critic_params":{
        "gnn_params":gnn_params,
        "mlp_params":mlp_params,
        "lr":5e-4,
        "decay_interval":100,
        "decay_rate":0.95,
        "device":shared_params["device"],
    },
    "uncertainty_critic_params":uncertainty_critic_params[shared_params["uncertainty_policy"]],
    "memory_params":{
        "memory_size":shared_params["memory_size"],
        "num_atoms":shared_params["num_atoms"],
        "time_to_live":500,
    },
    "env_params":{
        "if_trunc":True,
        "max_steps":10000,
        "fmax":0.0001,
    },
    "generator_params":{
        "num_atoms":shared_params["num_atoms"],
        "pos_scale":shared_params["pos_scale"],
        "action_scale":shared_params["action_scale"],
        "threshold":1.0,
        "action_policy":shared_params["epsilon_policy"],
    },
    "num_atoms":shared_params["num_atoms"],
    "alpha":2.0,
    "use_dataset":shared_params["use_dataset"],
    "dataset_path":data_file,
    "if_memory_update":shared_params["if_replace"],
    "if_greedy":shared_params["if_greedy"],
}

maxucb_params={

}