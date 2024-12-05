#!/bin/bash 
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G --time=1:29:00
#SBATCH -c 1 
#SBATCH --exclude=gpu[11-17] 
#SBATCH --output=outputs22/slurm_%A.out




cd protBERT
source protbert_venv/bin/activate
echo starting 



#python demo.py <path_to_data> <name_to_datadict> --base_dict nonexsisting.bin
 


