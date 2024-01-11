#!/bin/sh
env=''
num_runs=''
num_agents=''
wandb_name=''
wandb=false

while getopts e:r:n:i:w flag
do
    case "${flag}" in
        e) env=${OPTARG};;
        r) num_runs=${OPTARG};;
        n) num_agents=${OPTARG};;
        i) wandb_name=${OPTARG};;
        w) wandb=true;;
    esac
done

if [ ${wandb} = true ]
then
    echo "Env is ${env} with ${num_agents} agent(s) for ${num_runs} run(s). Using W&B for logging"
    python3 ../train/train_nashq_${env}.py --num_agents ${num_agents} --num_runs ${num_runs} --wandb --entity ${wandb_name}
else
    echo "Env is ${env} with ${num_agents} agent(s) for ${num_runs} run(s)"
    python3 ../train/train_nashq_${env}.py --num_agents ${num_agents} --num_runs ${num_runs} 
fi


