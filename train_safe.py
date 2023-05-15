import argparse
import os
from datetime import datetime
import numpy as np
import torch

from slac.algo import LatentPolicySafetyCriticSlac, SafetyCriticSlacAlgorithm
from slac.env import make_carla
from slac.trainer import Trainer
import json
from configuration import get_default_config

params = {
    'number_of_vehicles': 100,# number of surounding vehicles
    'number_of_walkers': 0,# number of surounding walkers
    'town': 'Town03', # which town to simulate
    'dt': 0.1, # time interval between two frames
    'continuous_accel_range': [-3.0, 3.0], # continuous acceleration range
    'continuous_steer_range': [-0.5, 0.5],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*', # filter for defining ego vehicle
    'port': 2000, # CARLA service's port
    'env_mode': 'train', # mode of env (collect/train/evaluate)
    'max_time_episode': 500, # maximum timesteps per episode
    'max_past_step': 1,
    'desired_speed': 20, # desired speed (m/s)
    'max_ego_spawn_times': 10, # maximum times to spawn ego vehicle
    'spectator': True, # spectator
    'obs_size': 256, # obs size pixel: [256 * 3, 256 * 2]
    'grid_size': [2, 3], # grid size: 
    'obs_range':40, # obs_range (meter)
    'd_behind': 8, # distance behind the ego vehicle (meter)
    'distances_from_start': 20, # norm distance from start to destination (m)
    
    # NOTE Task1:Roundabout
    'start': [52.1 , -4.2, 0.6, 0.0, 0.0, 178.66],# start waypoint
    'destination': [4.46, -61.46, 0.6, 0.0, 0.0, 0.0],# destination
    # NOTE Task2:Ramp merge
    # 'start': [-102.6, 22.6, 0.6,0.0, 0.0, 135],# start waypoint
    # 'destination': [-153, 102, 0.6,0.0, 0.0, 90],# destination
    # NOTE Task3:Intersection
    # 'start': [-35.0,-135.0, 0.6, 0.0, 0.0, 0.0],
    # 'destination': [10.7,-178, 0.6, 0.0, 0.0, -90.0],
    
    'enable_display': True , # enable display manager
    }


def main(args):
    config = get_default_config()
    config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["initial_collection_steps"] = args.initial_collection_steps
    config["initial_learning_steps"] = args.initial_learning_steps
    config["num_steps"] = args.num_steps
    config["eval_interval"] = args.eval_interval
    config["record_interval"] = args.record_interval
    config["num_eval_episodes"] = args.num_eval_episodes
    config["seed"] = args.seed
    
    env = make_carla(
        env_name= f"{config['domain_name']}-{config['task_name']}",
        params= params,
        action_repeat=config["action_repeat"],
        image_size=config["image_size"])

    env_test = env

    log_dir = os.path.join(
        "logs",
        "safe-slac-03",
        f"{config['domain_name']}-{config['task_name']}",
        f'slac-seed{config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )
    
    algo = LatentPolicySafetyCriticSlac(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=config["action_repeat"],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=config["seed"],
        gamma_c=config["gamma_c"],
        batch_size_sac=config["batch_size_sac"],
        batch_size_latent=config["batch_size_latent"],
        num_sequences=config["num_sequences"],
        buffer_size=config["buffer_size"],
        lr_sac=config["lr_sac"],
        lr_latent=config["lr_latent"],
        feature_dim=config["feature_dim"],
        z1_dim=config["z1_dim"],
        z2_dim=config["z2_dim"],
        hidden_units=config["hidden_units"],
        tau=config["tau"],
        start_alpha=config["start_alpha"],
        start_lagrange=config["start_lagrange"],
        grad_clip_norm=config["grad_clip_norm"],
        image_noise=config["image_noise"],
        budget=config["budget"],
        record_interval=config["record_interval"],
    )

    trainer = Trainer(
        num_sequences=config["num_sequences"],
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=config["seed"],
        num_steps=config["num_steps"],
        initial_learning_steps=config["initial_learning_steps"],
        initial_collection_steps=config["initial_collection_steps"],
        collect_with_policy=config["collect_with_policy"],
        eval_interval=config["eval_interval"],
        num_eval_episodes=config["num_eval_episodes"],
        action_repeat=config["action_repeat"],
        train_steps_per_iter=config["train_steps_per_iter"],
        env_steps_per_train_step=config["env_steps_per_train_step"],
    )

    trainer.writer.add_text("algo_config", json.dumps(config), 0)
    trainer.writer.add_text("env_parms", json.dumps(params), 0)
    
    #trainer.train()

    save_dir = "./models/model_merge"
    algo.load_model(save_dir)
    env_test.action_repeat = 4
    trainer.play(50)

    trainer.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, default="Carla", help="Name of the domain")
    parser.add_argument("--task_name", type=str, default="Sensors", help="Name of the task")
    parser.add_argument("--initial_collection_steps", type=int, default=5 * 10 **4, help="Number of initial collection steps")#4
    parser.add_argument("--initial_learning_steps", type=int, default=1 * 10 ** 4, help="Number of initial learning steps")#4
    parser.add_argument("--num_steps", type=int, default=3 * 10 ** 5, help="Number of training steps")
    parser.add_argument("--eval_interval", type=int, default=25 * 10 ** 3, help="Evaluate interval steps")#3
    parser.add_argument("--record_interval", type=int, default=1 * 10 ** 3, help="Record interval steps")#3
    parser.add_argument("--num_eval_episodes", type=int, default= 3, help="Number of evaluate episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda", type=bool,default=True, help="Train using GPU with CUDA")
    parser.add_argument("--play", type=bool,default=False, help="Evaluate the agent.")
    args = parser.parse_args()
    main(args)
