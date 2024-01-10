import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .policy import ReplayMemory, DQN, Transition

class Agent():
    
    def __init__(self, agent_id, obs_space, action_space, config):
        self.id = agent_id
        self.action_space = action_space
        self.obs_space = obs_space
        self.memory = ReplayMemory(1000000)
        self.policy_net = DQN(self.obs_space.shape, self.action_space.n).to(config['device'])
        self.target_net = DQN(self.obs_space.shape, self.action_space.n).to(config['device'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimiser = optim.AdamW(self.policy_net.parameters(), lr=config['LR'], amsgrad=True)
        self.rng = np.random.default_rng(12345)
    
    def select_action(self, config, state):
        sample = self.rng.random()
        if sample > config['eps threshold']:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).to(config['device'])).max(0)[1].view(1,1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=config['device'], dtype=torch.long)
        
    def select_action_optimal(self, config, state):
        with torch.no_grad():
            return self.policy_net(torch.from_numpy(state).to(config['device'])).max(0)[1].view(1,1)
    
    def optimise_model(self, config):
        if len(self.memory) < config['batch size']:
            return
        transitions = self.memory.sample(config['batch size'])
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # As we don't have any next states (s_{t+1} doesn't exist), we compute the expected state action values used in Hu et al. (2003) and Leibo et al. (2017)
        
        expected_state_action_values = (1 - config['alpha'])*state_action_values + config['alpha']*reward_batch
        
        # Compute the Huber loss
        loss = config['loss func'](state_action_values, expected_state_action_values)
        
        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*config['TAU'] + target_net_state_dict[key]*(1-config['TAU'])
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss
    
class NashAgent():
    
    def __init__(self, agent_id, alpha):
    
        self.agent_id = agent_id
        self.q_table = np.zeros((100,1))
        self.rng = np.random.default_rng(12345)
        self.alpha = alpha
        
    def select_action(self, config):
        
        sample = self.rng.random()
        if sample > config['eps_value']:
            xmax = np.argmax(self.q_table, axis=0)
            self.x_action = xmax
            return xmax
        else:
            self.x_action = np.random.randint(low=0, high=99, size=1)
            return self.x_action
        
    def update_policy(self, reward):
        self.q_table[self.x_action] = (1 - self.alpha)*self.q_table[self.x_action] + self.alpha*reward
    