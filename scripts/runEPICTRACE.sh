#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --mem=50G --time=70:00:00
#SBATCH -c 4  
#SBATCH --output=slurm_%A.out


source epictrace_venv/bin/activate
set -e
task=$((3-1)) # the task to run for default datasets 1 for TPP2 or 2 for TPP3
versioncode=123 #just three digit code to identify the run (900 for final models)
split=0 #the crossvaliadtaion split 0-9
n=0 #the cross-validation index 0-4

# D_alphabeta,beta = IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split}

#for Tpp3 task we recommend using --lr_s and --manual_SWA with defaults
# for TPP2 --lr 0.0001 and --manual_SWA with --SWA_max_lr 0.0001
#####
#base training
python src/train.py --max_epochs=80 -v ${versioncode}$(($n % 5))${task}${split} -d IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split} -i=Long_ievdj_mcpas_alpha.bin --lr_s 
# OR (creates protBERTembeddings automatically, in subsequent runs set -i to the path of the embeddings)
# python src/train.py --max_epochs=80 -v <versioncode> --train <path_to_train> --val <path_to_val> --test <path_to_test> --lr_s --manual_SWA
#SWA training (can be done jointly by adding --manual_SWA to base training)
python src/train.py --max_epochs=0 --load -v ${versioncode}$(($n % 5))${task}${split} -d IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split} --test_task ${task} -i=Long_ievdj_mcpas_alpha.bin --lr_s --manual_SWA

# SWA run identifier _c<lr>__<SWA_cycle>_<SWA_epochs>_<idx>
SWA_r=_c0.0001_1_20_01 

# predicting on  specified test set for a single split
python src/test_results.py ${versioncode}${n}${task} --runs ${split} --save_preds --SWA --SWA_run ${SWA_r} --dataset data/IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split}_tpp${task}_data.gz

# same but with specified path for preds
# python src/test_results.py ${versioncode}${n}${task} --runs ${split} --save_preds --SWA --SWA_run ${SWA_r} --dataset data/IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split}_tpp${task}_data.gz --pred_save_path testingsavepath.csv

# predicting on default named test set  for all splits
# equivalent to above but with all splits and dataset is implicitly specified (default name)

# splits=(0 1 2 3 4 5 6 7 8 9)
# python src/test_results.py ${versioncode}$(($n % 5))${task} --runs ${splits[*]} --save_preds --SWA --SWA_run ${SWA_r}

# if default name used (i.e. --dataset argument is not used) exdata is "" otherwise exdata defaults to full data path with '/' replaced by '_'
exdata="data_IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV_ab_b${split}_tpp${task}_data.gz"

#collect_PEpi_path uses the predicted 
python src/test_results.py ${versioncode}${n}${task} --collect_PEpi_path loggingDir22/folder22/version_${versioncode}${n}${task}${split}/checkpoints/${versioncode}${n}${task}${split}_manual_swa${SWA_r}${exdata}preds.csv
# Using with specified path for preds and specified save version
#python src/test_results.py ${versioncode}${n}${task} --collect_PEpi_path testingsavepath.csv --save_version 123456test --dataset=data/IEVDJcor310fPEpivalR_17_6$(($n % 5))tpp${task}_CV/ab_b${split}_tpp${task}_data.gz 



python src/test_results.py ${versioncode}${n}${task} --collect_path loggingDir22/folder22/version_${versioncode}${n}${task}${split}/checkpoints/${versioncode}${n}${task}${split}_manual_swa${SWA_r}${exdata}preds.csv --out_name extra_identifier

#####




# gather results from all runs to same files for analysis
python sgather _results.py
