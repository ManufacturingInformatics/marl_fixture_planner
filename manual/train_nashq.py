from algorithms.nashq.runner import Runner
import os

NUM_RUNS = 1

def main():
    
    for r in range(NUM_RUNS):

      print(f"---------- RUN {r+3} ----------")

      runner = Runner(run_num=r+3)
      runner.run()

if __name__ == "__main__":
    main()
