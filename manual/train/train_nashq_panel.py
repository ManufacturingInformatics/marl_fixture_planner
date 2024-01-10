from algorithms.nashq_panel.runner import Runner
import argparse
import os

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int)
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    for r in range(args.num_runs):

      print(f"---------- RUN {r+1} ----------")

      runner = Runner(run_num=r+1, num_agents=args.num_agents, wandb=args.wandb)
      runner.run()

if __name__ == "__main__":
    main()
