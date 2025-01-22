import sys, re, os, pickle
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------- #

# Fix Warmup Bug
from warmup_scheduler import (
    GradualWarmupScheduler,
)  # https://github.com/ildoonet/pytorch-gradual-warmup-lr


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


# ---------------------------------------------------------------------------- #


def WritePredictionToCsv(csv_out_name, img_array, prediction_array, ground_truth_array):
    # expect input_array[0] to have image name
    # @prediction_array sample_size x num_label

    fout = open(csv_out_name, "w")
    #
    temp1 = prediction_array.shape[1]

    try:  # ? if there is only 1 label, then ground_truth_array is not 2d matrix
        temp2 = ground_truth_array.shape[1]
    except:
        # https://stackoverflow.com/questions/12021754/how-to-slice-a-pandas-dataframe-by-position
        ground_truth_array = ground_truth_array[:, np.newaxis]
        temp2 = ground_truth_array.shape[1]

    column_header = (
        "image_name,"
        + ",".join("pred_label" + str(i) for i in range(temp1))
        + ","
        + ",".join("true_label" + str(i) for i in range(temp2))
        + "\n"
    )
    fout.write(column_header)

    # write out each line in the arrays
    for i in range(len(img_array)):
        fout.write(
            img_array[i]
            + ","
            + ",".join(str(x) for x in prediction_array[i])
            + ","
            + ",".join(str(x) for x in ground_truth_array[i])
            + "\n"
        )
    # end
    fout.close()
