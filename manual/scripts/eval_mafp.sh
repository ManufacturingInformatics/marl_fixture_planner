#!/bin/sh
env=''
num_runs=''
num_agents=''
run_name=''

while getopts e:r:n:a: flag
do
    case "${flag}" in
        e) env=${OPTARG};;
        r) num_runs=${OPTARG};;
        n) num_agents=${OPTARG};;
        a) run_name=${OPTARG};;
    esac
done

echo "Env is ${env} with ${num_agents} agent(s) for ${num_runs}. Using run ${run_name} for evaluation"
python3 ../train/eval_nashq_${env}.py --num_agents ${num_agents} --num_runs ${num_runs} --run_name ${run_name}