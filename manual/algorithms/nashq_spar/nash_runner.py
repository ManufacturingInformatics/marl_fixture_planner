import numpy as np
import torch
import math
from .environment import create_actions, create_contexts, MultiAgentFixtureBandit
from .agent import NashAgent
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

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3500
ALPHA = 0.8
TAU = 0.005
NUM_STEPS = 2500

class NashQRunner:
    
    def __init__(self, num_agents, run_num):
        
        self.run_num = run_num
        self.contexts = create_contexts()
        self.actions = create_actions()
        self.env = MultiAgentFixtureBandit(self.contexts, self.actions, num_agents=num_agents)
        
        self.config = {}
        self.config['steps done'] = 0
        self.config['eps_value'] = EPS_START
        
        self.rewards_file = f'./runs_csv/nash_q_proof/rewards_run_{self.run_num}.csv'
        
        self.deformation_file = f'./runs_csv/nash_q_proof/obs_run_{self.run_num}.csv'
        
        val = self.env.num_actions % self.env.num_agents
        if val == 0:
            num_split_actions = int(self.env.num_actions / self.env.num_agents)
        else:
            num_split_actions = int((self.env.num_actions - val) / self.env.num_agents)
        
        self.agents = {}
        self.agents_offset = {}
        
        for i in range(self.env.num_agents):
            agent_id = 'agent' + str(i)
            self.agents[agent_id] = NashAgent(agent_id, alpha=ALPHA)
            self.agents_offset[agent_id] = i*num_split_actions
            
        parent_dir = "./runs_csv"
        dir_name = "nash_q_proof" 
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path) is False:
            os.mkdir(path)
            
        with open(self.rewards_file, 'w') as f:
            writer = csv.writer(f)
            header = ['step', 'reward']
            writer.writerow(header)
            
        with open(self.deformation_file, 'w') as f:
            writer = csv.writer(f)
            header = ['step', 'x', 'y', 'z']
            writer.writerow(header)
            
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
    
    def run(self):
        
        start = time.time()
        
        self.config['eps threshold'] = self.decay_eps()
        
        for step in tqdm(range(NUM_STEPS), desc=f'Run {self.run_num}', colour='green'):
            
            actions = []
            actions_dict = {}
            
            for id, agent in self.agents.items():
                action = agent.select_action(self.config)
                action_offset = action + self.agents_offset[id]
                actions.append(action_offset)
                actions_dict[id] = action
                
            obs, reward, done, _ = self.env.step(actions, self.env.context_list[0])
            
            for id, agent in self.agents.items():
                agent.update_policy(reward)
                
            self.config['steps done'] = step
            self.config['eps_value'] = self.decay_eps()
            
            with open(self.deformation_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([step] + obs.flatten().tolist())
                
            with open(self.rewards_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([step, reward])
             
        end = time.time()
        fps = NUM_STEPS/(end - start)
        print(f"Complete. Average FPS: {fps} fps")
            
    def decay_eps(self):
        return EPS_END + (EPS_START - EPS_END)*math.exp(-1*(self.config['steps done']/EPS_DECAY))
    
    def test_fea(self):
        """
        Run the generated sample from MATLAB to ensure that the system is working
        """
        actions = []
        for id, agent in self.agents.items():
            action = agent.select_action(self.config)
            action_offset = action + self.agents_offset[id]
            actions.append(action_offset)
        obs, reward, done, _  = self.env.step(actions, self.env.context_list[0])
        return done