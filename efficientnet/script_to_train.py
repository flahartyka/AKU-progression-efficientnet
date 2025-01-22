import os
import pickle
import re
import sys
import time
from datetime import datetime

script = """#!/bin/bash

source myconda
#conda activate pytorch2
mamba activate base
module load CUDA/11.8 # ! cannot use cuda12 with apex
module load cuDNN/8.9.2/CUDA-11
module load gcc/11.3.0


# ---------------------------------------------------------------------------- #

export HF_HOME=/data/flahartyka/huggingface/
export TRANSFORMERS_CACHE=/data/flahartyka/huggingface/

main_data_dir=/data/flahartyka/Multilabel # ! this is your image csv directory

batch_size=64 

run_name=RUN_NAME # ! this is experiment name

model_dir=$main_data_dir/PredictVacuum/$run_name
mkdir $model_dir

log_dir=$main_data_dir/PredictVacuum/$run_name

cd /data/flahartyka/Alkaptonuria/Example/Manual # ! this is where you keep the code

# --------------------------------------------------------------------------#

image_size=448

# ---------------------------------------------------------------------------- #

# ! train

#image_csv=/data/flahartyka/Multilabel/multilabel_vacuum_lumbar_ground_truth_train.csv

#python3 train_and_eval_multilabel.py --image_csv $image_csv --run_name $run_name --image_size $image_size --model_arch_name tf_efficientnet_b4_ns --model_dir $model_dir --log_dir $log_dir --fold FOLD --num_ground_truth_label 20 --n_epochs 50 --batch_size $batch_size --init_lr LEARNING_RATE --use_amp

# --use_amp

# ---------------------------------------------------------------------------- #

# ! test

#image_csv=/data/flahartyka/Multilabel/multilabel_vacuum_lumbar_ground_truth_test.csv
#load_this_model=/data/flahartyka/Multilabel/PredictVacuum/lumbar-vacuum-multilabel/lumbar-vacuum-multilabel_best_val_loss_fold0.pth

#python3 train_and_evaluate.py --image_csv $image_csv --run_name $run_name --image_size $image_size --model_arch_name tf_efficientnet_b4_ns --model_dir $model_dir --log_dir $log_dir --num_ground_truth_label 3  --batch_size $batch_size --load_this_model $load_this_model --run_on_test_img 


#python3 train_and_eval_multilabel.py --image_csv $image_csv --run_name $run_name --image_size $image_size --model_arch_name tf_efficientnet_b4_ns --model_dir $model_dir --log_dir $log_dir --num_ground_truth_label 20 --batch_size $batch_size --load_this_model $load_this_model --run_on_test_img --attribution_method Occlusion 
#--attribution_start_idx 3 --attribution_end_idx 5



"""

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

path = "/data/flahartyka/Multilabel/PredictVacuum"
os.chdir(path)

counter = 0

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

for learn_rate in [0.00003]:  # 0.00001,
    # ! use time stamp to make main model folder
    # ! all folds will be saved in this same folder.
    now = datetime.now()
    script2 = re.sub("RUN_NAME", "lumbar-vacuum-multilabel", script)
    #
    for fold in [0]: # 1, 2, 3, 4
        script2 = re.sub("FOLD", str(fold), script2)
        script2 = re.sub("LEARNING_RATE", str(learn_rate), script2)
        now = datetime.now()  # current date and time
        scriptname = (
            "script" + str(counter) + "-" + now.strftime("%m-%d-%H-%M-%S") + ".sh"
        )
        fout = open(scriptname, "w")
        fout.write(script2)
        fout.close()
        #
        time.sleep(1)
        #os.system('sbatch --partition=gpu --time=4:00:00 --gres=gpu:v100x:2 --mem=30g --cpus-per-task=16 ' + scriptname )
        #os.system('sh ' + scriptname )
        # os.system(
        #     "sbatch --partition=gpu --time=08:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=8 "
        #     + scriptname
        # )
        # os.system('sbatch --time=1:00:00 --mem=16g --cpus-per-task=12 ' + scriptname )
        counter = counter + 1


exit()

