# Import Required External Libraries
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# Import Required Internal Libraries
from _20_model import dqn


class ReplayDataset(Dataset):
    def __init__(self, max_size, device):
        """================================================================================================
        ## Initialization for Replay Dataset
        ================================================================================================"""
        self.max_size = int(max_size)
        self.device = device
        self.transitions = []

    def append(self, state, action, reward, state_next, done):
        """================================================================================================
        ## Append Transition Tensor to Replay Dataset
        ================================================================================================"""
        action_idx = int(torch.argmax(
            torch.as_tensor(action, dtype=torch.float32)).item())
        self.transitions.append(
            (
                torch.as_tensor(state, dtype=torch.float32),
                torch.tensor([action_idx], dtype=torch.long),
                torch.tensor(float(reward), dtype=torch.float32),
                torch.as_tensor(state_next, dtype=torch.float32),
                torch.tensor(float(done), dtype=torch.float32),
            )
        )

        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def __len__(self):
        """================================================================================================
        ## Return Dataset Length
        ================================================================================================"""
        return len(self.transitions)

    def __getitem__(self, index):
        """================================================================================================
        ## Return Transition by Index
        ================================================================================================"""
        return self.transitions[index]


class Dqn:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Parameters for DQN
        ================================================================================================"""
        # - Load Parameter Sets
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        # - Parameters for Epsilon-Greedy Policy
        self.epsilon = float(self.train_conf["epsilon_start"])
        self.epsilon_end = float(self.train_conf["epsilon_end"])
        self.epsilon_decay = float(self.train_conf["epsilon_decay"])

        # - Parameters for Calculating Return
        self.gamma = float(self.train_conf["gamma"])

        # - Device for Training
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # - Learning Rate for Training Networks
        self.learning_rate = float(self.train_conf["learning_rate"])

        # - Dimensions for Neural Network
        self.state_dim = int(dqn._03_state_design.get_state_dim())
        self.dim_action = len(dqn._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        # - Parameters for Replay Buffer
        self.replay_buffer_size = int(self.train_conf["replay_buffer_size"])
        self.replay_buffer = ReplayDataset(
            self.replay_buffer_size, self.device)
        self.replay_start_size = int(self.train_conf["replay_start_size"])
        self.batch_size = int(self.train_conf["batch_size"])

        # - Parameter for Target Network Update Interval
        self.target_update_interval = int(
            self.train_conf["target_update_interval"])

        # - Initial Values for Training
        self.training_steps_init = int(self.train_conf["training_steps_init"])
        self.training_steps = self.training_steps_init
        self.loss_function = nn.MSELoss()

        self.update_every = int(self.train_conf["update_every"])
        self.env_steps = 0

        """================================================================================================
        ## Load Target Policy
        ================================================================================================"""
        # - Target Policy Name
        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        # - Path for Target Policy
        self.policy_path = os.path.join(
            self.conf.path_dqn_policy,
            self.policy_name + '.pth')

        # - Create Policy Network
        self.policy = dqn._02_network.create_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)

        # - Load Existing Weights if Present
        if os.path.exists(self.policy_path):
            if self.conf.train_rewrite is not True:
                self.policy.load_state_dict(torch.load(
                    self.policy_path,
                    map_location=self.device,
                    weights_only=True,
                ))

        # - Create Target Policy by Copying Policy Network
        self.target_policy = dqn._02_network.create_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

        # - Create Optimizer for Policy Network
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.learning_rate)

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        # Map State Material to Designed State
        state = self.map_to_designed_state(state_mat)

        # Action Selection by Epsilon-Greedy Policy
        action_mat = dqn._06_algorithm.epsilon_greedy_action_selection(
            policy=self.policy, state=state, epsilon=self.epsilon)
        action = self.map_to_designed_action(action_mat)

        # Run Environment and Get Transition
        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        # Map to Designed State and Reward
        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        # Aggregate Transition
        transition = (state, action_mat, state_next, reward_next, done, score)

        # Update Epsilon for Next Action Selection
        self.update_epsilon()

        # Return Transition
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Update Q-Network by Transition
        ================================================================================================"""
        # - Unpack Transition
        state, action, state_next, reward_next, done, _ = transition

        # - Append Transition to Replay Buffer
        self.replay_buffer.append(
            state, action, reward_next, state_next, done)

        # - Skip Update Until Replay Buffer is Warmed Up
        if len(self.replay_buffer) < self.replay_start_size:
            return
    
        if self.env_steps % self.update_every != 0:
            self.env_steps += 1
            return
        self.env_steps += 1

        # - Build Replay DataLoader
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer.transitions, batch_size)

        states, actions, rewards, states_next, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        states_next = torch.stack(states_next).to(self.device)
        dones = torch.stack(dones).to(self.device)

        # - Calculate Predicted Q-Values
        qvalues = self.policy(states).gather(1, actions).squeeze(1)

        # - Calculate Target Q-Values
        qvalues_next = self.target_policy(states_next)\
            .max(dim=1).values.detach()
        qtargets = rewards + self.gamma * qvalues_next * (1.0 - dones)

        # - Optimize Policy Network
        loss = self.loss_function(qvalues, qtargets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # - Synchronize Target Network Periodically
        if self.training_steps % self.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        self.training_steps += 1

    def get_train_configuration(self):
        train_conf = dqn._01_params.get_train_params()
        return train_conf

    def update_epsilon(self):
        """====================================================================================================
        ## Get next epsilon by Algorithm
        ===================================================================================================="""
        # Calculate next epsilon by decay
        self.epsilon = dqn._06_algorithm.\
            decay_epsilon(epsilon_start=self.epsilon, epsilon_decay=self.epsilon_decay,
                          epsilon_end=self.epsilon_end)

    def map_to_designed_state(self, state_mat):
        """====================================================================================================
        ## Mapping from Environment State to Designed State
        ===================================================================================================="""
        state_custom = dqn._03_state_design.calculate_state_key(
            state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        """====================================================================================================
        ## Mapping from Policy Action to Designed Action
        ===================================================================================================="""
        action_custom = action_mat *\
            dqn._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        """====================================================================================================
        ## Mapping from Environment Reward to Designed Reward
        ===================================================================================================="""
        reward_custom = dqn._05_reward_design.calculate_reward(
            reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        """====================================================================================================
        ## Select Action for Playing
        ===================================================================================================="""
        state = self.map_to_designed_state(state_mat)
        action_mat = dqn._06_algorithm.epsilon_greedy_action_selection(
            policy=self.policy, state=state, epsilon=epsilon)
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        dqn._02_network.save_nn(self.policy, self.policy_path)
