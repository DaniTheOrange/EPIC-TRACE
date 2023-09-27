#!/bin/bash

#SBATCH --time=3:30:00 -c 1 --mem=6G
#SBATCH --output=outputs22/split_%A_%a.out
##SBATCH --array=0-4



source epictrace_venv/bin/activate




# for n in 1 2 3 4 ; do
# n=$SLURM_ARRAY_TASK_ID

n=0
python split.py data/IEVDJcor410fPEpivalR_17_6_alphabeta${n} -d data/VDJDB_to_split_2022-06-17_corrected.csv data/IEDB_to_split_2022-06-17_corrected.csv --tpp_task 1 --folds 10 --neg_multiplier 5 --per_epi --use_discarded --similarity_subset CDR3 V J Epitope , alpha aV aJ Epitope --TCR_tsim CDR3 , alpha --require_paired
# python split.py data/IEVDJ10fPEpivalR_17_6_AB${n} -d data/VDJDB_to_split_2022-06-17.csv data/IEDB_to_split_2022-06-17.csv --tpp_task 1 --folds 10 --neg_multiplier 5 --per_epi --use_discarded --require_paired



# done