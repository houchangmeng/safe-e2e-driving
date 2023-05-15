import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from slac.network.initializer import initialize_weight
from slac.utils import build_mlp, calculate_kl_divergence


class FixedGaussian(nn.Module):#torch.jit.ScriptModule
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, output_dim, std):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    #@torch.jit.script_method
    def forward(self, x):
        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Gaussian, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    #@torch.jit.script_method
    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std

class Bernoulli(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super(Bernoulli, self).__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.ELU(),
        ).apply(initialize_weight)

    #@torch.jit.script_method
    def forward(self, x):
        if x.ndim == 3:
            B, S, _ = x.size()
            x = self.net(x.view(B * S, _)).view(B, S, -1)
        else:
            x = self.net(x)
        p = torch.sigmoid(x)
        return p


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, input_dim=288, output_dim=3, std=1.0):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        ).apply(initialize_weight)
        self.std = std

    #@torch.jit.script_method
    def forward(self, x):
        B, S, latent_dim = x.size()
        x = x.view(B * S, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x, torch.ones_like(x).mul_(self.std)

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, input_dim=3, output_dim=256):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.ELU(inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ELU(inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ELU(inplace=True),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.ELU(inplace=True),
        ).apply(initialize_weight)

    #@torch.jit.script_method
    def forward(self, x):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x = self.net(x)
        x = x.view(B, S, -1)
        return x


class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super(LatentModel, self).__init__()
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )
        
        # feat(t) = Encoder(x(t))
        self.camera_encoder = Encoder(state_shape[0], feature_dim)
        self.lidar_encoder = Encoder(state_shape[0], feature_dim)
        self.birdeye_encoder = Encoder(state_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.camera_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(0.1))
        self.lidar_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(0.1))
        self.birdeye_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(0.1))

        self.apply(initialize_weight)
    
    def get_features(self,state):
        camera_feature = self.camera_encoder(state['camera'])
        lidar_feature = self.lidar_encoder(state['lidar'])
        birdeye_feature = self.lidar_encoder(state['birdeye'])

        return (camera_feature + lidar_feature) * 0.5

    #@torch.jit.script_method
    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    #@torch.jit.script_method
    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    #@torch.jit.script_method
    def calculate_loss(self, state_, action_, reward_, done_):
        # Calculate the sequence of features.
        feature_ = self.get_features(state_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        ## Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)

        # Prediction loss of camera.
        state_mean_, state_std_ = self.camera_decoder(z_)
        state_noise_ = (state_["camera"] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_camera = -log_likelihood_.mean(dim=0).sum()

        # Prediction loss of lidar.
        state_mean_, state_std_ = self.lidar_decoder(z_)
        state_noise_ = (state_['lidar'] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_lidar = -log_likelihood_.mean(dim=0).sum()

        # Prediction loss of birdeye.
        state_mean_, state_std_ = self.birdeye_decoder(z_)
        state_noise_ = (state_['birdeye'] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_birdeye = -log_likelihood_.mean(dim=0).sum()

        loss_image = loss_camera + loss_lidar + loss_birdeye

        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()
        return loss_kld, loss_image, loss_reward
    
    #@torch.jit.script_method
    def sample_latent(self,state_, action_):
        # Sample from latent variable model.

        # Calculate the sequence of features.
        features_ = self.get_features(state_)

        # Posterior  
        # q(z1(0) | feat(0))
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1_posterior = z1_mean + torch.randn_like(z1_std) * z1_std
        # q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_posterior)
        z2_posterior = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_post = [z1_posterior]
        z2_post = [z2_posterior]

        # Prior
        # p(z1(0)) = N(0, I)
        z1_mean_, z1_std_ = self.z1_prior_init(action_[:, 0])
        z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
        # p(z2(0) | z1(0)) = q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_prior)
        z2_prior = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_pri = [z1_prior]
        z2_pri = [z2_prior]

        # Conditional Prior
        # q(z1(0) | feat(0))
        z1_condition = z1_posterior
        # p(z2(0) | z1(0))
        z2_condition = z2_posterior
        z1_cond = [z1_condition]
        z2_cond = [z2_condition]

        for t in range(1, action_.size(1) + 1):
            # Posterior
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2_posterior, action_[:, t - 1]], dim=1))
            z1_posterior = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1_posterior, z2_posterior, action_[:, t - 1]], dim=1))
            z2_posterior = z2_mean + torch.randn_like(z2_std) * z2_std
            z1_post.append(z1_posterior)
            z2_post.append(z2_posterior)

            # Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_prior, action_[:, t - 1]], dim=1)) 
            z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_, z2_std_ = self.z2_prior(torch.cat([z1_prior, z2_prior, action_[:, t - 1]], dim=1))
            z2_prior = z2_mean_ + torch.randn_like(z2_std_) * z2_std_
            z1_pri.append(z1_prior)
            z2_pri.append(z2_prior)

            # Conditional Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_cond, z1_std_cond = self.z1_prior(torch.cat([z2_condition, action_[:, t - 1]], dim=1)) 
            z1_condition = z1_mean_cond + torch.randn_like(z1_std_cond) * z1_std_cond
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_cond, z2_std_cond = self.z2_prior(torch.cat([z1_condition, z2_condition, action_[:, t - 1]], dim=1))
            z2_condition = z2_mean_cond + torch.randn_like(z2_std_cond) * z2_std_cond
            z1_cond.append(z1_condition)
            z2_cond.append(z2_condition)

        z1_post = torch.stack(z1_post, dim=1)
        z2_post = torch.stack(z2_post, dim=1)

        z1_pri = torch.stack(z1_pri, dim=1)
        z2_pri = torch.stack(z2_pri, dim=1)

        z1_cond = torch.stack(z1_cond, dim=1)
        z2_cond = torch.stack(z2_cond, dim=1)

        # Prediction loss of images.
        z_post = torch.cat([z1_post, z2_post], dim=-1)
        z_cond = torch.cat([z1_cond, z2_cond], dim=-1)
        z_pri = torch.cat([z1_pri, z2_pri], dim=-1)
        
        # Ground Truth
        ground_truth = torch.cat([state_['camera'], state_['lidar'], state_['birdeye']], dim=-1) # chw
        
        # Posterior Sample
        camera_post, _ = self.camera_decoder(z_post)
        lidar_post, _ = self.lidar_decoder(z_post)
        birdeye_post, _ = self.birdeye_decoder(z_post)
        post = torch.cat([camera_post,lidar_post,birdeye_post], dim=-1) 

        # Conditional Prior Sample
        camera_pri_cond, _ = self.camera_decoder(z_cond)
        lidar_pri_cond, _ = self.lidar_decoder(z_cond)
        birdeye_pri_cond, _ = self.birdeye_decoder(z_cond)
        pri_cond = torch.cat([camera_pri_cond, lidar_pri_cond, birdeye_pri_cond], dim=-1) 

        # Prior Sample
        camera_pri, _ = self.camera_decoder(z_pri)
        lidar_pri, _ = self.lidar_decoder(z_pri)
        birdeye_pri, _ = self.birdeye_decoder(z_pri)
        pri = torch.cat([camera_pri,lidar_pri,birdeye_pri], dim=-1) 

        idx = torch.randint(low=0, high=32,size=(1,))
        img_concat = torch.cat([ground_truth[idx].view(-1, 3, 64, 192), 
                                post[idx].view(-1, 3, 64, 192),
                                pri_cond[idx].view(-1, 3, 64, 192),
                                pri[idx].view(-1, 3, 64, 192)], dim=0)

        return img_concat

class CostLatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics, reward and cost.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        image_noise=0.1
    ):
        super(CostLatentModel, self).__init__()
        self.bceloss = torch.nn.BCELoss(reduction="none")
        # p(z1(0)) = N(0, I)
        self.z1_prior_init = FixedGaussian(z1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.z2_prior_init = Gaussian(z1_dim, z2_dim, hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = Gaussian(
            z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_prior = Gaussian(
            z1_dim + z2_dim + action_shape[0],
            z2_dim,
            hidden_units,
        )

        # q(z1(0) | feat(0))
        self.z1_posterior_init = Gaussian(feature_dim, z1_dim, hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_posterior = Gaussian(
            feature_dim + z2_dim + action_shape[0],
            z1_dim,
            hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2_posterior = self.z2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )

        self.cost = Gaussian(
            2 * z1_dim + 2 * z2_dim + action_shape[0],
            1,
            hidden_units,
        )
        
        # feat(t) = Encoder(x(t))
        self.camera_encoder = Encoder(state_shape[0], feature_dim)
        self.lidar_encoder = Encoder(state_shape[0], feature_dim)
        self.birdeye_encoder = Encoder(state_shape[0], feature_dim)
        # p(x(t) | z1(t), z2(t))
        self.camera_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(image_noise))
        self.lidar_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(image_noise))
        self.birdeye_decoder = Decoder(z1_dim + z2_dim,state_shape[0],std=np.sqrt(image_noise))
        
        self.apply(initialize_weight)

    #@torch.jit.script_method
    def get_features(self,state):
        camera_feature = self.camera_encoder(state['camera'])
        lidar_feature = self.lidar_encoder(state['lidar'])
        birdeye_feature = self.lidar_encoder(state['birdeye'])

        return (camera_feature + lidar_feature) * 0.5

    #@torch.jit.script_method
    def sample_prior(self, actions_, z2_post_):
        # p(z1(0)) = N(0, I)
        z1_mean_init, z1_std_init = self.z1_prior_init(actions_[:, 0])
        # p(z1(t) | z2(t-1), a(t-1))
        z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_post_[:, : actions_.size(1)], actions_], dim=-1))
        # Concatenate initial and consecutive latent variables
        z1_mean_ = torch.cat([z1_mean_init.unsqueeze(1), z1_mean_], dim=1)
        z1_std_ = torch.cat([z1_std_init.unsqueeze(1), z1_std_], dim=1)
        return (z1_mean_, z1_std_)

    #@torch.jit.script_method
    def sample_posterior(self, features_, actions_):
        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1 = z1_mean + torch.randn_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1)
        z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_mean_ = [z1_mean]
        z1_std_ = [z1_std]
        z1_ = [z1]
        z2_ = [z2]

        for t in range(1, actions_.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2, actions_[:, t - 1]], dim=1))
            z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1, z2, actions_[:, t - 1]], dim=1))
            z2 = z2_mean + torch.randn_like(z2_std) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = torch.stack(z1_mean_, dim=1)
        z1_std_ = torch.stack(z1_std_, dim=1)
        z1_ = torch.stack(z1_, dim=1)
        z2_ = torch.stack(z2_, dim=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    # @torch.jit.script_method
    def calculate_loss(self, state_, action_, reward_, done_, cost_):
        # Calculate the sequence of features.
        feature_ = self.get_features(state_)

        # Sample from latent variable model.
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_posterior(feature_, action_)
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(action_, z2_)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(dim=0).sum()

        # Prediction loss of images.
        z_ = torch.cat([z1_, z2_], dim=-1)
        # Prediction loss of camera.
        state_mean_, state_std_ = self.camera_decoder(z_)
        state_noise_ = (state_['camera'] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_camera = -log_likelihood_.mean(dim=0).sum()

        # Prediction loss of lidar.
        state_mean_, state_std_ = self.lidar_decoder(z_)
        state_noise_ = (state_['lidar'] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_lidar = -log_likelihood_.mean(dim=0).sum()
        
        # Prediction loss of birdeye.
        state_mean_, state_std_ = self.birdeye_decoder(z_)
        state_noise_ = (state_['birdeye'] - state_mean_) / (state_std_ + 1e-8)
        log_likelihood_ = (-0.5 * state_noise_.pow(2) - state_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_birdeye = -log_likelihood_.mean(dim=0).sum()

        loss_image = loss_camera + loss_lidar + loss_birdeye

        # Prediction loss of rewards.
        x = torch.cat([z_[:, :-1], action_, z_[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean_, reward_std_ = self.reward(x.view(B * S, X))
        reward_mean_ = reward_mean_.view(B, S, 1)
        reward_std_ = reward_std_.view(B, S, 1)
        reward_noise_ = (reward_ - reward_mean_) / (reward_std_ + 1e-8)
        log_likelihood_reward_ = (-0.5 * reward_noise_.pow(2) - reward_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_reward = -log_likelihood_reward_.mul_(1 - done_).mean(dim=0).sum()

        cost_mean_, cost_std_ = self.cost(x.view(B * S, X))
        cost_mean_ = cost_mean_.view(B, S, 1)
        cost_std_ = cost_std_.view(B, S, 1)
        cost_noise_ = (cost_ - cost_mean_) / (cost_std_ + 1e-8)
        log_likelihood_cost_ = (-0.5 * cost_noise_.pow(2) - cost_std_.log()) - 0.5 * math.log(2 * math.pi)
        loss_cost = -log_likelihood_cost_.mul_(1 - done_).mean(dim=0).sum()
        
        return loss_kld, loss_image, loss_reward, loss_cost
    
    #@torch.jit.script_method
    def sample_latent(self,state_, action_):
        # Sample from latent variable model.

        # Calculate the sequence of features.
        features_ = self.get_features(state_)

        # Posterior  
        # q(z1(0) | feat(0))
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1_posterior = z1_mean + torch.randn_like(z1_std) * z1_std
        # q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_posterior)
        z2_posterior = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_post = [z1_posterior]
        z2_post = [z2_posterior]

        # Prior
        # p(z1(0)) = N(0, I)
        z1_mean_, z1_std_ = self.z1_prior_init(action_[:, 0])
        z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
        # p(z2(0) | z1(0)) = q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_prior)
        z2_prior = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_pri = [z1_prior]
        z2_pri = [z2_prior]

        # Conditional Prior
        # q(z1(0) | feat(0))
        z1_condition = z1_posterior
        # p(z2(0) | z1(0))
        z2_condition = z2_posterior
        z1_cond = [z1_condition]
        z2_cond = [z2_condition]

        for t in range(1, action_.size(1) + 1):
            # Posterior
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_posterior(torch.cat([features_[:, t], z2_posterior, action_[:, t - 1]], dim=1))
            z1_posterior = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2_posterior(torch.cat([z1_posterior, z2_posterior, action_[:, t - 1]], dim=1))
            z2_posterior = z2_mean + torch.randn_like(z2_std) * z2_std
            z1_post.append(z1_posterior)
            z2_post.append(z2_posterior)

            # Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_prior, action_[:, t - 1]], dim=1)) 
            z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_, z2_std_ = self.z2_prior(torch.cat([z1_prior, z2_prior, action_[:, t - 1]], dim=1))
            z2_prior = z2_mean_ + torch.randn_like(z2_std_) * z2_std_
            z1_pri.append(z1_prior)
            z2_pri.append(z2_prior)

            # Conditional Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_cond, z1_std_cond = self.z1_prior(torch.cat([z2_condition, action_[:, t - 1]], dim=1)) 
            z1_condition = z1_mean_cond + torch.randn_like(z1_std_cond) * z1_std_cond
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_cond, z2_std_cond = self.z2_prior(torch.cat([z1_condition, z2_condition, action_[:, t - 1]], dim=1))
            z2_condition = z2_mean_cond + torch.randn_like(z2_std_cond) * z2_std_cond
            z1_cond.append(z1_condition)
            z2_cond.append(z2_condition)

        z1_post = torch.stack(z1_post, dim=1)
        z2_post = torch.stack(z2_post, dim=1)

        z1_pri = torch.stack(z1_pri, dim=1)
        z2_pri = torch.stack(z2_pri, dim=1)

        z1_cond = torch.stack(z1_cond, dim=1)
        z2_cond = torch.stack(z2_cond, dim=1)

        # Prediction loss of images.
        z_post = torch.cat([z1_post, z2_post], dim=-1)
        z_cond = torch.cat([z1_cond, z2_cond], dim=-1)
        z_pri = torch.cat([z1_pri, z2_pri], dim=-1)
        
        # Ground Truth
        ground_truth = torch.cat([state_['camera'], state_['lidar'], state_['birdeye']], dim=-1) # chw
        
        # Posterior Sample
        camera_post, _ = self.camera_decoder(z_post)
        lidar_post, _ = self.lidar_decoder(z_post)
        birdeye_post, _ = self.birdeye_decoder(z_post)
        post = torch.cat([camera_post,lidar_post,birdeye_post], dim=-1) 

        # Conditional Prior Sample
        camera_pri_cond, _ = self.camera_decoder(z_cond)
        lidar_pri_cond, _ = self.lidar_decoder(z_cond)
        birdeye_pri_cond, _ = self.birdeye_decoder(z_cond)
        pri_cond = torch.cat([camera_pri_cond, lidar_pri_cond, birdeye_pri_cond], dim=-1) 

        # Prior Sample
        camera_pri, _ = self.camera_decoder(z_pri)
        lidar_pri, _ = self.lidar_decoder(z_pri)
        birdeye_pri, _ = self.birdeye_decoder(z_pri)
        pri = torch.cat([camera_pri,lidar_pri,birdeye_pri], dim=-1) 

        idx = torch.randint(low=0, high=32,size=(1,))
        img_concat = torch.cat([ground_truth[idx].view(-1, 3, 64, 192), 
                                post[idx].view(-1, 3, 64, 192),
                                pri_cond[idx].view(-1, 3, 64, 192),
                                pri[idx].view(-1, 3, 64, 192)], dim=0)
        
        return img_concat
    
    def sample_prior_latent(self,state_, action_):
        # Sample from latent variable model.

        # Prior
        # p(z1(0)) = N(0, I)
        z1_mean_, z1_std_ = self.z1_prior_init(action_[:, 0])
        z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
        # p(z2(0) | z1(0)) = q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_prior)
        z2_prior = z2_mean + torch.randn_like(z2_std) * z2_std

        z1_pri = [z1_prior]
        z2_pri = [z2_prior]

        for t in range(1, action_.size(1) + 1):
           
            # Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_, z1_std_ = self.z1_prior(torch.cat([z2_prior, action_[:, t - 1]], dim=1)) 
            z1_prior = z1_mean_ + torch.randn_like(z1_std_) * z1_std_
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_, z2_std_ = self.z2_prior(torch.cat([z1_prior, z2_prior, action_[:, t - 1]], dim=1))
            z2_prior = z2_mean_ + torch.randn_like(z2_std_) * z2_std_
            z1_pri.append(z1_prior)
            z2_pri.append(z2_prior)

        z1_pri = torch.stack(z1_pri, dim=1)
        z2_pri = torch.stack(z2_pri, dim=1)

        return z1_pri, z2_pri
    
    def sample_cond_latent(self,state_, action_):
        # Sample from latent variable model.

        # Calculate the sequence of features.
        features_ = self.get_features(state_)

        # Posterior  
        # q(z1(0) | feat(0))
        z1_mean, z1_std = self.z1_posterior_init(features_[:, 0])
        z1_posterior = z1_mean + torch.randn_like(z1_std) * z1_std
        # q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_posterior_init(z1_posterior)
        z2_posterior = z2_mean + torch.randn_like(z2_std) * z2_std

        # Conditional Prior
        # q(z1(0) | feat(0))
        z1_condition = z1_posterior
        # p(z2(0) | z1(0))
        z2_condition = z2_posterior
        z1_cond = [z1_condition]
        z2_cond = [z2_condition]

        for t in range(1, action_.size(1) + 1):

            # Conditional Prior
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean_cond, z1_std_cond = self.z1_prior(torch.cat([z2_condition, action_[:, t - 1]], dim=1)) 
            z1_condition = z1_mean_cond + torch.randn_like(z1_std_cond) * z1_std_cond
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean_cond, z2_std_cond = self.z2_prior(torch.cat([z1_condition, z2_condition, action_[:, t - 1]], dim=1))
            z2_condition = z2_mean_cond + torch.randn_like(z2_std_cond) * z2_std_cond
            z1_cond.append(z1_condition)
            z2_cond.append(z2_condition)

        z1_cond = torch.stack(z1_cond, dim=1)
        z2_cond = torch.stack(z2_cond, dim=1)

        return z1_cond, z2_cond