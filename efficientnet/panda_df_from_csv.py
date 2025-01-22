import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from glob import glob

# ---------------------------------------------------------------------------- #

main_dir = "/data/flahartyka/Multilabel"

dir_on_server = "/data/flahartyka/Multilabel/Cervical"

csv = os.path.join(main_dir, "C2-C3_narrowing.csv")
df = pd.read_csv(csv)
df = df.dropna(subset=["Number"]).reset_index(drop=True)


fout = os.path.join(main_dir, "narrowing_c2c3_ground_truth_train.csv")
fout = open(fout, "w")
fout.write("img_path,image_name,ground_truth_label\n")

image_folder = os.path.join(main_dir, "Cervical")


for index, row in df.iterrows():
    # ! 3 labels
    label = np.zeros(3)
    # shift -1 because python index, scoring was done at 1,2,3,4 scale
    label[int(row["Curvature"] - 1)] = 1
    label = ";".join(str(s) for s in label)

    # ! image name is listed strangely in the csv.
    # need to reconstruct the name
    # 0001 0 35 --> 0001035.png
    image_name = (
        "{0:0=4d}".format(int(row["Number"]))
        + str(int(row["Gender"]))
        + str(int(row["Age"]))
        + ".png"
    )

    # ! do we need other measurements?

    # ! check image path exist
    img_path = os.path.join(image_folder, image_name)
    if not os.path.exists(img_path):
        print(img_path)
        continue

    # define this wrt to server path. 
    # img_path = os.path.join(dir_on_server, image_name) # ! very weird if we run on window
    img_path = dir_on_server+'/'+image_name # ! 
    
    # ! output
    fout.write(",".join([img_path, image_name, label]) + "\n")


#
fout.close()

# ---------------------------------------------------------------------------- #
# ! add fold assignment below.

csv = os.path.join(main_dir, "curvature_ground_truth.csv")
df = pd.read_csv(csv)

# shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# ! make 6 folds, 5 will be use in 5-fold cross validation, the 6th one is test set.
assign_fold = np.random.choice(np.arange(6), size=len(df))
df["fold"] = assign_fold

# !
df_temp = df[df["fold"] != 5].reset_index(drop=True)
df_temp.to_csv(
    os.path.join(main_dir, "calcium_l4-l5_ground_truth_train.csv"), index=None
)
print("total sample size", len(df_temp))

# !
df_temp = df[df["fold"] == 5].reset_index(drop=True)
df_temp.to_csv(
    os.path.join(main_dir, "curvature_ground_truth_test.csv"), index=None
)
print("total sample size", len(df_temp))
