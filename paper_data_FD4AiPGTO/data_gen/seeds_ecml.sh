#!/bin/bash 
 
#SBATCH --job-name=ECML_data
#SBATCH --output=seeds_ecml.txt
#SBATCH --partition=CLUSTER
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=120
#SBATCH --time=7-12
#SBATCH --gpus-per-node=0
#SBATCH --oversubscribe
#SBATCH --mem=200GB
#SBATCH --propagate=STACK
 
eval "$(conda shell.bash hook)"
conda activate py310


cd ~/AI2Go/Paper

echo "Done activating Conda Env. Start get_seed"

python -m get_seed_ecml -s 4483 &
python -m get_seed_ecml -s 2120 &
python -m get_seed_ecml -s 6825 &
python -m get_seed_ecml -s 5612 &
python -m get_seed_ecml -s 2224 &

wait
echo "First 5 Seeds done"

python -m get_seed_ecml -s 4735 &
python -m get_seed_ecml -s 1016 &
python -m get_seed_ecml -s 3891 &
python -m get_seed_ecml -s 2377 &
python -m get_seed_ecml -s 4582 &

wait
echo "First 10 Seeds done"
