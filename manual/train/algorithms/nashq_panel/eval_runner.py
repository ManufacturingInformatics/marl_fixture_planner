import numpy as np
import torch
import math
from .environment import create_actions, create_contexts, MultiAgentFixtureBandit
from .agent import Agent
from tqdm import tqdm
import time
from time import sleep
import csv
from datetime import datetime
from gym import spaces
import wandb
import argparse
import os
import traceback

NUM_STATES = 500
NUM_EPISODES = 100
BATCH_SIZE = 32
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3500
ALPHA = 0.8
TAU = 0.005
LR = 1e-4

class Evaluator:
    
    def __init__(self, **kwargs):
        
        self.config = {}
        self.run_name = kwargs['run_name']
        self.num_runs = kwargs['num_runs']
        self.config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config['batch size'] = BATCH_SIZE
        self.config['LR'] = LR
        self.config['alpha'] = ALPHA
        self.config['TAU'] = TAU
        self.config['steps done'] = 0
        self.config['eps threshold'] = EPS_START
        print(f"Using device {self.config['device']}")
        
        self.contexts = create_contexts()
        np.random.shuffle(self.contexts)
        self.actions = create_actions()
        self.env = MultiAgentFixtureBandit(self.contexts, self.actions, num_agents=kwargs['num_agents'])
        
        self.reward_storage = f'../runs_csv/wing_panel/marl_{self.env.num_agents}_agents/evaluation/rewards.csv'
        
        self.action_storage = f'../runs_csv/wing_panel/marl_{self.env.num_agents}_agents/evaluation/actions.csv'
        
        self.context_storage = f'../runs_csv/wing_panel/marl_{self.env.num_agents}_agents/evaluation/contexts.csv'
        
        parent_dir = "../runs_csv/wing_panel"
        dir_name = f"marl_{str(self.env.num_agents)}_agents" 
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path) is False:
            os.mkdir(path)
        
        run_dir = 'evaluation'
        path = os.path.join(path, run_dir)
        if os.path.isdir(path) is False:
            os.mkdir(path)
        
        self.agents = {}
        self.agents_offset = {}
        
        val = self.env.num_actions % self.env.num_agents
        if val == 0:
            num_split_actions = int(self.env.num_actions / self.env.num_agents)
            self.agent_action_space = spaces.Discrete(num_split_actions)
        else:
            num_split_actions = int((self.env.num_actions - val) / self.env.num_agents)
            self.agent_action_space = spaces.Discrete(num_split_actions)
        
        for i in range(self.env.num_agents):
            agent_id = 'agent' + str(i)
            self.agents[agent_id] = Agent(agent_id, self.env.contexts, self.agent_action_space, self.config)
            self.agents[agent_id].policy_net.load_state_dict(torch.load(f'../train/agent_weights/wing_panel/marl_{self.env.num_agents}_agents/run_{self.run_name}/agent_{agent_id}_dict.pt', map_location=self.config['device']))
            self.agents_offset[agent_id] = i*num_split_actions
            
        try:
            if self.test_fea():
                a = """
                          Surrogate model is functioning. 
                    ===========================================
                    """
                print(a)
        except Exception:
            print("Error with MATLAB engine:")
            print(traceback.format_exc())
            raise SystemExit(0)    
    
    def evaluate(self):
        
        start = time.time()
        
        for run in range(self.num_runs):
            
            run_start = time.time()
            
            postfix = {
                'total reward': 0.0,
                'total regret': 0.0
            }
            
            for step in tqdm(range(self.env.num_contexts), desc=f'Run {run+1}', colour='green'):
                
                actions = []
                actions_dict = {}
                
                state = self.env.context_list[step]
                
                for id, agent in self.agents.items():
                    action = agent.select_action_optimal(self.config, state)
                    action_offset = action.cpu().detach().numpy() + self.agents_offset[id]
                    actions.append(action_offset)
                    actions_dict[id] = action
                    
                obs, reward, done, _ = self.env.step(actions, state)
                postfix['total reward'] += reward
                
                with open(self.reward_storage, 'a', encoding='UTF8') as f:
                    csv_file = csv.writer(f)
                    csv_file.writerow([step, reward, obs[0,0], obs[0,1], obs[0,2]])
                
                action_pos = []
                for i in range(self.env.num_agents):
                    action_pos.append(self.actions[actions[i], :].tolist())
                    
                with open(self.action_storage, 'a', encoding='UTF8') as f:
                    csv_file = csv.writer(f)
                    csv_file.writerow([action_pos])
                    
                with open(self.context_storage, 'a', encoding='UTF8') as f:
                    csv_file = csv.writer(f)
                    csv_file.writerow(state)
                
            postfix['total regret'] = self.env.num_agents*self.env.num_contexts*self.env.mean_reward - postfix['total reward']*self.env.num_agents
            
            run_end = time.time()
            print(f"Epsiode {run+1}/{self.num_runs}, FPS: {self.env.num_contexts/(run_end-run_start)} fps")
            print(f"Epsiode Reward: {postfix['total reward']}, Episode Regret: {postfix['total regret']}\n")
            
        end = time.time()
        print(f"Complete. Total time: {end - start} seconds")
    
    def test_fea(self):
        """
        Run the generated sample from MATLAB to ensure that the system is working
        """
        actions = []
        for id, agent in self.agents.items():
            action = agent.select_action_optimal(self.config, self.env.context_list[0])
            action_offset = action.cpu().detach().numpy() + self.agents_offset[id]
            actions.append(action_offset)
        obs, reward, done, _  = self.env.step(actions, self.env.context_list[0])
        return done

