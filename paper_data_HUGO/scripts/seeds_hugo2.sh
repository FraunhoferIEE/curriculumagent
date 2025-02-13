#!/bin/bash 
 
#SBATCH --job-name=TopologyAgent​ 
#SBATCH --output=seed_paralllel.txt​ 
#SBATCH --partition=CLUSTER
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=22
#SBATCH --time=7-12
#SBATCH --gpus-per-node=0
#SBATCH --oversubscribe
#SBATCH --mem=200GB
#SBATCH --propagate=STACK
 
eval "$(conda shell.bash hook)"
conda activate py310


cd ~/AI2Go/Paper

echo "Done activating Conda Env. Start get_seed"

python -m get_seed_hugo_parallel2 -s 4483 &
python -m get_seed_hugo_parallel2 -s 2120 &
python -m get_seed_hugo_parallel2 -s 6825 &
python -m get_seed_hugo_parallel2 -s 5612 &
python -m get_seed_hugo_parallel2 -s 2224 &

wait
echo "First 5 Seeds done"

python -m get_seed_hugo_parallel2 -s 4735 &
python -m get_seed_hugo_parallel2 -s 1016 &
python -m get_seed_hugo_parallel2 -s 3891 &
python -m get_seed_hugo_parallel2 -s 2377 &
python -m get_seed_hugo_parallel2 -s 4582 &

wait
echo "First 10 Seeds done"

python -m get_seed_hugo_parallel2 -s 3484 &
python -m get_seed_hugo_parallel2 -s 2015 &
python -m get_seed_hugo_parallel2 -s 1501 &
python -m get_seed_hugo_parallel2 -s 6987 &
python -m get_seed_hugo_parallel2 -s 7768 &

wait
echo "15 Seeds done"

python -m get_seed_hugo_parallel2 -s 2069 &
python -m get_seed_hugo_parallel2 -s 5229 &
python -m get_seed_hugo_parallel2 -s 7503 &
python -m get_seed_hugo_parallel2 -s 2809 &
python -m get_seed_hugo_parallel2 -s 5715 &

echo " 20 Seeds are done"
