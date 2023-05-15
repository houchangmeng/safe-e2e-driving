from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = {}
        self.state_['camera'] = deque(maxlen=self.num_sequences + 1)
        self.state_['lidar'] = deque(maxlen=self.num_sequences + 1)
        self.state_['birdeye'] = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_['camera'].clear()
        self.state_['lidar'].clear()
        self.state_['birdeye'].clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_['camera'].append(state['camera'])
        self.state_['lidar'].append(state['lidar'])
        self.state_['birdeye'].append(state['camera'])

    def append(self, action, reward, done, next_state):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_['camera'].append(next_state['camera'])
        self.state_['lidar'].append(next_state['lidar'])
        self.state_['birdeye'].append(next_state['birdeye'])

    def get(self):
        state_ = {}
        state_['camera'] = LazyFrames(self.state_['camera'])
        state_['lidar'] = LazyFrames(self.state_['lidar'])
        state_['birdeye'] = LazyFrames(self.state_['birdeye'])
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = {}
        self.state_['camera']  = [None] * buffer_size
        self.state_['lidar']   = [None] * buffer_size
        self.state_['birdeye'] = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action_ = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = SequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.append(action, reward, done, next_state)

        if self.buff.is_full():
            state_, action_, reward_, done_ = self.buff.get()
            self._append(state_, action_, reward_, done_)

        if episode_done:
            self.buff.reset()

    def _append(self, state_, action_, reward_, done_):
        self.state_['camera'][self._p] = state_['camera']
        self.state_['lidar'][self._p] = state_['lidar']
        self.state_['birdeye'][self._p] = state_['birdeye']

        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = {}
        state_['camera'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['lidar'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['birdeye'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_['camera'][i, ...] = self.state_['camera'][idx]
            state_['lidar'][i, ...] = self.state_['lidar'][idx]
            state_['birdeye'][i, ...] = self.state_['birdeye'][idx]
        state_['camera'] = torch.tensor(state_['camera'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['lidar'] = torch.tensor(state_['lidar'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['birdeye'] = torch.tensor(state_['birdeye'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = {}
        state_['camera'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['lidar'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['birdeye'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_['camera'][i, ...] = self.state_['camera'][idx]
            state_['lidar'][i, ...] = self.state_['lidar'][idx]
            state_['birdeye'][i, ...] = self.state_['birdeye'][idx]
        state_['camera'] = torch.tensor(state_['camera'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['lidar'] = torch.tensor(state_['lidar'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['birdeye'] = torch.tensor(state_['birdeye'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n

class CostSequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = {}
        self.state_['camera'] = deque(maxlen=self.num_sequences + 1)
        self.state_['lidar'] = deque(maxlen=self.num_sequences + 1)
        self.state_['birdeye'] = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)
        self.cost_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_['camera'].clear()
        self.state_['lidar'].clear()
        self.state_['birdeye'].clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()
        self.cost_.clear()

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_['camera'].append(state['camera'])
        self.state_['lidar'].append(state['lidar'])
        self.state_['birdeye'].append(state['camera'])

    def append(self, action, reward, done, next_state, cost):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.cost_.append([cost])
        self.state_['camera'].append(next_state['camera'])
        self.state_['lidar'].append(next_state['lidar'])
        self.state_['birdeye'].append(next_state['birdeye'])

    def get(self):
        state_ = {}
        state_['camera'] = LazyFrames(self.state_['camera'])
        state_['lidar'] = LazyFrames(self.state_['lidar'])
        state_['birdeye'] = LazyFrames(self.state_['birdeye'])
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        cost_ = np.array(self.cost_, dtype=np.float32)
        return state_, action_, reward_, done_, cost_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class CostReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = {}
        self.state_['camera']  = [None] * buffer_size
        self.state_['lidar']   = [None] * buffer_size
        self.state_['birdeye'] = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action_ = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.cost_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = CostSequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done, cost):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.append(action, reward, done, next_state, cost)

        if self.buff.is_full():
            state_, action_, reward_, done_, cost_ = self.buff.get()
            self._append(state_, action_, reward_, done_, cost_)

        if episode_done:
            self.buff.reset()

    def _append(self, state_, action_, reward_, done_, cost_):
        self.state_['camera'][self._p] = state_['camera']
        self.state_['lidar'][self._p] = state_['lidar']
        self.state_['birdeye'][self._p] = state_['birdeye']

        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))
        self.cost_[self._p].copy_(torch.from_numpy(cost_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = {}
        state_['camera'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['lidar'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['birdeye'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_['camera'][i, ...] = self.state_['camera'][idx]
            state_['lidar'][i, ...] = self.state_['lidar'][idx]
            state_['birdeye'][i, ...] = self.state_['birdeye'][idx]
        state_['camera'] = torch.tensor(state_['camera'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['lidar'] = torch.tensor(state_['lidar'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['birdeye'] = torch.tensor(state_['birdeye'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes], self.cost_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = {}
        state_['camera'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['lidar'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        state_['birdeye'] = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_['camera'][i, ...] = self.state_['camera'][idx]
            state_['lidar'][i, ...] = self.state_['lidar'][idx]
            state_['birdeye'][i, ...] = self.state_['birdeye'][idx]
        state_['camera'] = torch.tensor(state_['camera'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['lidar'] = torch.tensor(state_['lidar'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        state_['birdeye'] = torch.tensor(state_['birdeye'], dtype=torch.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1], self.cost_[idxes, -1]

    def __len__(self):
        return self._n
