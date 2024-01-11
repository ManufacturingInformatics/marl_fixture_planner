from algorithms.nashq_panel.eval_runner import Evaluator
import argparse
import os

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int)
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('--run_name')
    args = parser.parse_args()
    
    evaluator = Evaluator(num_agents=args.num_agents, num_runs=args.num_runs, run_name=args.run_name)
    evaluator.evaluate()
    
    
if __name__ == "__main__":
    main()