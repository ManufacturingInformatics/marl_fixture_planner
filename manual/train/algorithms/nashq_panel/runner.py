import numpy as np
import torch
import math
from .environment import create_actions, create_contexts, MultiAgentFixtureBandit
from .agent import Agent
from tqdm import tqdm
import time
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

class Runner:
    
    def __init__(self, **kwargs):
        
        self.config = {}
        self.run_num = kwargs['run_num']
        self.wandb = kwargs['wandb']
        self.config['loss func'] = torch.nn.SmoothL1Loss()
        self.config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.config['device']}")
        self.config['batch size'] = BATCH_SIZE
        self.config['LR'] = LR
        self.config['alpha'] = ALPHA
        self.config['TAU'] = TAU
        self.config['steps done'] = 0
        self.config['eps threshold'] = self.decay_eps()
        self.contexts = create_contexts()
        self.actions = create_actions()
        self.env = MultiAgentFixtureBandit(self.contexts, self.actions, num_agents=kwargs['num_agents'])
        self.td_file = f'../runs_csv/wing_panel/marl_{self.env.num_agents}_agents/run_{self.run_num}/td_loss_{datetime.now().strftime("%d-%m-%Y")}.csv'
        self.rewards_file = f'../runs_csv/wing_panel/marl_{self.env.num_agents}_agents/run_{self.run_num}/rewards_{datetime.now().strftime("%d-%m-%Y")}.csv'
        
        if self.wandb:
            wandb_config = {
                "learning rate": self.config['LR'],
                "alpha": self.config['alpha'],
                "batch_size": self.config['batch size'],
                "num_agents": self.env.num_agents
            }
            
            wandb.init(
                project="rl-fixture-tracking", 
                entity=args.wandb_name,
                name=f"mafd_agents{self.env.num_agents}_run{self.run_num}",
                config=wandb_config
            )
            
            wandb.define_metric("episode")
            wandb.define_metric("metrics/return", step_metric="episode")
            wandb.define_metric("metrics/regret", step_metric="episode")
        
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
            self.agents_offset[agent_id] = i*num_split_actions
            
        parent_dir = "../runs_csv/wing_panel"
        dir_name = f"marl_{str(self.env.num_agents)}_agents" 
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path) is False:
            os.mkdir(path)
        
        run_dir = f"run_{self.run_num}"
        path = os.path.join(path, run_dir)
        os.mkdir(path)
            
        with open(self.td_file, 'w+') as f:
            writer = csv.writer(f)
            header = ['step']
            for i in range(self.env.num_agents):
                header.append("td_loss_agent{}".format(i))
            writer.writerow(header)
        
        with open(self.rewards_file, 'w+') as f:
            writer = csv.writer(f)
            header = ['episode', 'total_reward', 'total_return']
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
        
        for i_episode in range(NUM_EPISODES):
            
            ep_start = time.time()
            
            self.config['eps threshold'] = self.decay_eps()
            
            a = "{0:.3g}".format(self.config['eps threshold'])
            
            postfix = {
                'total reward': 0.0,
                'episode regret': 0.0
            }
    
            for step in tqdm(range(self.env.num_contexts), desc=f'Episode {i_episode+1} | Learning Rate {a}', colour='green'):
                
                actions = []
                actions_dict = {}
                
                state = self.env.context_list[step]
                
                for id, agent in self.agents.items():
                    action = agent.select_action(self.config, state)
                    action_offset = action.cpu().detach().numpy() + self.agents_offset[id]
                    actions.append(action_offset)
                    actions_dict[id] = action
                
                obs, reward, done, _ = self.env.step(actions, state)
                # self.writer.add_scalar("step_reward/train", scalar_value=reward, global_step=self.config['steps done'])
                if self.wandb:
                    wandb.log(
                        {
                            "train/step_reward": reward
                        },
                        step=self.config['steps done']
                    )
                
                agent_losses = []
                agent_losses.append(self.config['steps done'])
                
                for id, agent in self.agents.items():
                    agent.memory.push(
                        torch.reshape(torch.from_numpy(state), (1,2)).to(self.config['device']), 
                        actions_dict[id], 
                        torch.reshape(torch.from_numpy(reward), (1,1)).to(self.config['device'])
                    )
                    loss = agent.optimise_model(self.config)
                    if loss is not None:
                        agent_losses.append(loss.cpu().detach().numpy())
                        # self.writer.add_scalar("train/{}_loss".format(id), scalar_value=loss, global_step=self.config['steps done'])
                        if self.wandb:
                            wandb.log(
                                {"loss/{}_loss".format(id): loss},
                                step=self.config['steps done']
                            )
                
                with open(self.td_file, 'a', encoding='UTF8') as f:
                    csv_file = csv.writer(f)
                    csv_file.writerow(agent_losses)                   
                
                postfix['total reward'] += reward
                    
                self.config['steps done'] += 1
                
            postfix['episode regret'] = self.env.num_agents*self.env.num_contexts*self.env.mean_reward - postfix['total reward']*self.env.num_agents
            
            if self.wandb:
                log_dict = {
                    "episode": i_episode,
                    "metrics/return": postfix['total reward'],
                    "metrics/regret": postfix['episode regret']
                }
            
                wandb.log(log_dict)
            
            with open(self.rewards_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([i_episode, postfix['total reward'], postfix['episode regret']])
                        
            ep_end = time.time()
            print(f"Epsiode {i_episode+1}/{NUM_EPISODES}, FPS: {self.env.num_contexts/(ep_end-ep_start)} fps")
            print(f"Epsiode Reward: {postfix['total reward']}, Episode Regret: {postfix['episode regret']}\n")
            
        end = time.time()
        print(f"Complete. Total time: {end - start} seconds")
        
        # Saving the models at the end of a run
        for id, agent in self.agents.items():
            torch.save(agent.policy_net.state_dict(), f'./runs_csv/marl_{self.env.num_agents}_agents/run_{self.run_num}/agent_{id}_dict.pt')
    
    def decay_eps(self):
        return EPS_END + (EPS_START - EPS_END)*math.exp(-1*(self.config['steps done']/EPS_DECAY))
    
    def test_fea(self):
        """
        Run the generated sample from MATLAB to ensure that the system is working
        """
        actions = []
        for id, agent in self.agents.items():
            action = agent.select_action(self.config, self.env.context_list[0])
            action_offset = action.cpu().detach().numpy() + self.agents_offset[id]
            actions.append(action_offset)
        obs, reward, done, _  = self.env.step(actions, self.env.context_list[0])
        return done