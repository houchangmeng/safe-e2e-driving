import os
import cv2
import torch
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm
from slac.utils import sample_reproduction_sensors


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences
        self._state = {}

    def reset_episode(self, state):
        self._state.clear()
        self._state['camera'] = deque(maxlen=self.num_sequences)
        self._state['lidar'] = deque(maxlen=self.num_sequences)
        self._state['birdeye'] = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state['camera'].append(np.zeros(self.state_shape, dtype=np.uint8))
            self._state['lidar'].append(np.zeros(self.state_shape, dtype=np.uint8))
            self._state['birdeye'].append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state['camera'].append(state['camera'])
        self._state['lidar'].append(state['lidar'])
        self._state['birdeye'].append(state['birdeye'])

    def append(self, state, action):
        self._state['camera'].append(state['camera'])
        self._state['lidar'].append(state['lidar'])
        self._state['birdeye'].append(state['birdeye'])
        self._action.append(action)

    @property
    def state(self):
        s = {
            'camera': np.array(self._state['camera'])[None, ...],
            'lidar': np.array(self._state['lidar'])[None, ...],
            'birdeye': np.array(self._state['birdeye'])[None, ...]}
        return s
    
    @property
    def last_state(self):
        s = {
            'camera': np.array(self._state['camera'][-1])[None, ...],
            'lidar': np.array(self._state['lidar'][-1])[None, ...],
            'birdeye': np.array(self._state['birdeye'][-1])[None, ...]}
        return s

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)

    @property
    def last_action(self):
        return np.array(self._action[-1])


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=2* 10 ** 4,
        initial_learning_steps=10 ** 4,
        collect_with_policy=False,
        num_sequences=8,
        eval_interval=1*10 ** 3,
        num_eval_episodes=3,
        env_steps_per_train_step=1,
        action_repeat=1,
        train_steps_per_iter=1
    ):
        # Env to collect samples.
        self.env = env
        #self.env.seed(seed)
        self.train_steps_per_iter = train_steps_per_iter
        # Env for evaluation.
        self.env_test = env_test
        #self.env_test.seed(2 ** 31 - seed)

        np.random.seed(seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": [], "cost": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir, flush_secs=10)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.env_steps_per_train_step=env_steps_per_train_step
        self.collect_with_policy = collect_with_policy

        # Time to start training.
        self.start_time = time()

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state , info = self.env.reset()

        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        print("# Collect trajectories using random policy.")
        bar = tqdm(range(1, self.initial_collection_steps + 1))
        for step in bar:
            bar.set_description("Collect trajectories using random policy.")
            t = self.algo.step(self.env, self.ob, t, (not self.collect_with_policy) and step <= self.initial_collection_steps, self.writer)
        
        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        print("# Updating latent variable model.")
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)
       
        # Iterate collection, update and evaluation.
        print("# Updating latent , actor, critic, safe critic, model.")
        train_latent_sac = True
        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            t = self.algo.step(self.env, self.ob, t, False, self.writer)
            self.algo.update_lag(t, self.writer)

            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)
            
            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
                self.algo.save_model(self.model_dir)
            
            if step_env % 1000 == 0:
                for sched in self.algo.scheds:
                    sched.step()

            if step_env % self.algo.record_interval == 0:
                self.writer.add_scalar("cost/train", np.mean(self.algo.epoch_costreturns), global_step=step_env)
                self.writer.add_scalar("return/train", np.mean(self.algo.epoch_rewardreturns), global_step=step_env)
                self.algo.epoch_costreturns = [0]
                self.algo.epoch_rewardreturns = [0]

        # Wait for logging to be finished.
        sleep(10)
    
    def debug_save_obs(self, state, name, step=0):
        self.writer.add_image(f"observation_{name}", state.astype(np.uint8), global_step=step)

    def evaluate(self, step_env):
        reward_returns = []
        cost_returns = []
        steps_until_dump_obs = 20
        def coord_to_im_(coord):
            coord = (coord+1.5)*100
            return coord.astype(int)
        
        obs_list = []
        recons_list = []
        video_spf = 4//self.action_repeat
        video_fps = 25/video_spf

        for i in range(self.num_eval_episodes):
            self.algo.z1 = None
            self.algo.z2 = None

            state, info  = self.env_test.reset()
            self.ob_test.reset_episode(state)

            episode_return = 0.0
            cost_return = 0.0
            eval_step = 0
            while True:
    
                action = self.algo.exploit(self.ob_test)
                
                if i == 0 and eval_step % video_spf == 0:

                    im_camera = self.ob_test.state["camera"][0][-1]
                    im_lidar  = self.ob_test.state["lidar"][0][-1]
                    im_birdeye = self.ob_test.state["birdeye"][0][-1]
                    im = np.concatenate([im_camera, im_lidar, im_birdeye],axis=2).astype("uint8")
                    obs_list.append(im)

                    rec_camera, rec_lidar, rec_birdeye = sample_reproduction_sensors(
                        self.algo.latent, 
                        self.algo.device,
                        self.ob_test.state, 
                        np.array([self.ob_test._action]))
                    rec_camera = rec_camera[0][-1]
                    rec_lidar = rec_lidar[0][-1]
                    rec_birdeye = rec_birdeye[0][-1]
                    reconstruction = np.concatenate([rec_camera, rec_lidar, rec_birdeye],axis=2)
                    reconstruction *= 255
                    reconstruction = reconstruction.astype("uint8")
                    
                    recons_list.append(reconstruction)
                   
                if steps_until_dump_obs == 0:
                    im_camera = self.ob_test.state["camera"][0][-1]
                    im_lidar  = self.ob_test.state["lidar"][0][-1]
                    im_birdeye = self.ob_test.state["birdeye"][0][-1]
                    im = np.concatenate([im_camera, im_lidar, im_birdeye],axis=2).astype("uint8")
                    
                    rec_camera, rec_lidar, rec_birdeye = sample_reproduction_sensors(
                        self.algo.latent, 
                        self.algo.device,
                        self.ob_test.state, 
                        np.array([self.ob_test._action]))
                    rec_camera = rec_camera[0][-1]
                    rec_lidar = rec_lidar[0][-1]
                    rec_birdeye = rec_birdeye[0][-1]
                    reconstruction = np.concatenate([rec_camera, rec_lidar, rec_birdeye],axis=2)
                    reconstruction *= 255
                    reconstruction = reconstruction.astype("uint8")

                    eval_img = np.concatenate([im, reconstruction],axis=1)
                    self.debug_save_obs(eval_img, "original_vs_reconstruction", step_env)
                steps_until_dump_obs -= 1

                state, reward, done, trunc, info = self.env_test.step(action)
                cost = info["cost"]
                
                self.ob_test.append(state, action)
                episode_return += reward
                cost_return += cost
                eval_step += 1                
                
                row_segment = np.ones((3, 64, 2)) * 255
                
                img = np.concatenate([
                    state['camera'],
                    row_segment,
                    state['lidar'],
                    row_segment,
                    state['birdeye']],axis=2) / 255.0
                rec_camera, rec_lidar, rec_birdeye = sample_reproduction_sensors(
                        self.algo.latent, 
                        self.algo.device,
                        self.ob_test.state, 
                        np.array([self.ob_test._action]))
                rec_camera = rec_camera[0][-1]
                rec_lidar = rec_lidar[0][-1]
                rec_birdeye = rec_birdeye[0][-1]
                rec = np.concatenate([
                    rec_camera,
                    row_segment,
                    rec_lidar,
                    row_segment, 
                    rec_birdeye],axis=2)

                img_size = 128
                col_segment = np.ones((3, 2, 64*3+2*2)) * 255
                img_concat = np.concatenate([img, col_segment, rec], axis=1)

                img_concat = img_concat.transpose((1,2,0))[:,:,::-1]
                cv2.imshow('Playback',cv2.resize(img_concat,(img_size*3+4, img_size*2+2)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if done or trunc or eval_step > self.env._max_episode_steps:
                    break

            if i==0:
                self.writer.add_video(f"video/eval", [np.concatenate([obs_list,recons_list], axis=2)], global_step=step_env, fps=video_fps)
            reward_returns.append(episode_return)
            cost_returns.append(cost_return)

        # Exit opencv
        cv2.destroyAllWindows()

        self.algo.z1 = None
        self.algo.z2 = None

        # Log to CSV.
        self.log["step"].append(step_env)
        mean_reward_return = np.mean(reward_returns)
        mean_cost_return = np.mean(cost_returns)
        median_reward_return = np.median(reward_returns)
        median_cost_return = np.median(cost_return)
        self.log["return"].append(mean_reward_return)
        self.log["cost"].append(mean_cost_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_reward_return, step_env)
        self.writer.add_scalar("return/test_median", median_reward_return, step_env)
        self.writer.add_histogram("return/test_hist", np.array(reward_returns), step_env)
        self.writer.add_scalar("cost/test", mean_cost_return, step_env)
        self.writer.add_scalar("cost/test_median", median_cost_return, step_env)
        self.writer.add_histogram("cost/test_hist", np.array(cost_returns), step_env)
        
        print(f"Steps: {step_env:<6}   " f"Return: {mean_reward_return:<5.1f} " f"CostRet: {mean_cost_return:<5.1f}   " f"Time: {self.time}")

    def play(self, num_play_episodes=1):

        for i in range(num_play_episodes):

            self.algo.z1 = None
            self.algo.z2 = None

            state, info  = self.env_test.reset()
            self.ob_test.reset_episode(state)

            while True:
                
                action = self.algo.exploit(self.ob_test) #exploit
                state, reward, done, trunc, info = self.env_test.step(action)
                self.ob_test.append(state, action)

                row_segment = np.ones((3, 64, 2)) * 255
                img = np.concatenate([
                    state['camera'],
                    row_segment,
                    state['lidar'],
                    row_segment,
                    state['birdeye']],axis=2) / 255.0
                rec_camera, rec_lidar, rec_birdeye = sample_reproduction_sensors(
                        self.algo.latent, 
                        self.algo.device,
                        self.ob_test.state, 
                        np.array([self.ob_test._action]))
                rec_camera = rec_camera[0][-1]
                rec_lidar = rec_lidar[0][-1]
                rec_birdeye = rec_birdeye[0][-1]
                rec = np.concatenate([
                    rec_camera,
                    row_segment,
                    rec_lidar,
                    row_segment, 
                    rec_birdeye],axis=2)

                img_size = 128
                col_segment = np.ones((3, 2, 64*3+2*2)) * 255
                img_concat = np.concatenate([img, col_segment, rec ], axis=1)

                img_concat = img_concat.transpose((1,2,0))[:,:,::-1]
                cv2.imshow('Playback',cv2.resize(img_concat,(img_size*3+4, img_size*2+2)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if done or trunc:
                    break

        # Exit opencv
        cv2.destroyAllWindows()

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
