import argparse
import json
import os
import random
import time

import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
# https://stackoverflow.com/questions/42480111/how-do-i-print-the-model-summary-in-pytorch
from torchinfo import summary

from models_upgrade import XrayClassifier
from sklearn.metrics import balanced_accuracy_score


from tqdm import tqdm

import util
from util import GradualWarmupSchedulerV2

from PIL import Image
import get_saliency

os.environ['HF_HOME'] = '/data/flahartyka/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/data/flahartyka/huggingface/'

# https://github.com/NVIDIA/apex/pull/1282
# https://github.com/NVIDIA/apex/issues/1735
#import apex # ! very hard to install, maybe don't use it?
# from apex import amp # ! https://github.com/NVIDIA/apex?tab=readme-ov-file#linux


from dataset import DatasetFromPandaDf, get_transforms

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_balanced_accuracy_score(prediction, target):
    return balanced_accuracy_score(target, prediction.argmax(axis=1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="MonthDateYear")
    parser.add_argument("--image_csv", type=str, default=None)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--model_arch_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--init_lr", type=float, default=3e-5)
    parser.add_argument("--num_ground_truth_label", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_meta", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--model_dir", type=str, default="./weights")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4")
    parser.add_argument("--n_meta_dim", type=str, default="512,128")

    # ! added some other useful args

    parser.add_argument("--seed", type=float, default=0)
    parser.add_argument(
        "--n_test",
        type=int,
        default=1,
        help="how many times do we flip images, 1=>no_flip, max=8",
    )
    parser.add_argument("--scheduler_scaler", type=float, default=10)
    parser.add_argument("--dropout_rate_in_forward", type=float, default=0.5)
    parser.add_argument("--no_scheduler", action="store_true", default=False)
    parser.add_argument(
        "--weights_in_loss",
        type=str,
        default=None,
        help="string input, weights on each kind of label, '1,5,10,1...'",
    )
    parser.add_argument("--load_this_model", type=str, default=None)
    parser.add_argument(
        "--run_on_test_img",
        action="store_true",
        default=False,
        help="run only on test set",
    )
    parser.add_argument(
        "--csv_out_name",
        type=str,
        default=None,
        help="csv with prediction on test set",
    )
    parser.add_argument(
        "--attribution_method",
        type=str,
        default=None,
        help="run attribution method",
    ) 
    parser.add_argument(
        "--attribution_start_idx",
        type=int,
        default=None,
        help="img index to start attribution",
    )
    parser.add_argument(
        "--attribution_end_idx",
        type=int,
        default=None,
        help="img index to end attribution",
    )
    
    # args, new_args = parser.parse_known_args()
    args = parser.parse_args()  # ! safer
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, criterion, args=None):
    
    # ! set gradscaler for amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    model.train()  # ! turn on dropout modules inside the model

    train_loss = []  # ! track the loss after looping over each batch.

    # loop over the dataloader, note: the loader returns 3 items: data, torch.tensor(label_np).long(), img_path
    bar = tqdm(loader)

    # ! loop over each batch (e.g. batch size 32) until end of data.
    for data, target, img_path in bar:
        
        if args.use_meta:  # ! if we want to add in age/gender/or whatever
            data, meta, target = data.to(device), meta.to(device), target.to(device)
        else:
            data, target = data.to(device), target.to(device)
            meta = None

        # fit the model on the data, gets the raw score as logits
        # logits = model(data, x_meta=meta)
        # if isinstance(logits, tuple):  # maybe we can return a bunch of things?
        #     logits, _ = logits

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            logits = model(data, x_meta=meta)
            loss = criterion(logits, target) # ! loss function
        #
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()  # ! reset optimizer to not keep gradient from previous batch.
        
        # if not args.use_amp:
        #     loss.backward()
        # else:
        #     # a few years ago, amp was not exactly part of pytorch. maybe we need to edit this.
        #     with amp.scale_loss(loss, optimizer) as scaled_loss: # ! use amp https://github.com/pytorch/pytorch/blob/main/docs/source/notes/amp_examples.rst
        #         scaled_loss.backward()

        # if args.image_size in [896, 576]:
        #     # ! stabilize training a bit by constraining gradient to be 0.5 max
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # ! send the loss back to CPU, because @train_loss is not on GPU
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)

        # smooth out loss with running average
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))

    # over all the batches within 1 epoch, see the averaged training loss
    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:  # ! return original if I = 0
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_one_epoch(model, loader, also_get_output=False, criterion=None, args=None):
    model.eval()  # ! turn off dropout

    n_test = args.n_test

    val_loss = []
    logits_array = []
    probs_array = []
    ground_truth_array = []
    img_path_array = []

    with torch.no_grad():  # ! not keeping gradient, will reduce GPU usage.
        # ! @loader is a dataloader on validation (or test) set
        for data, target, img_path in tqdm(loader):
            if args.use_meta:
                data, meta, target = data.to(device), meta.to(device), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
                meta = None

            
            # ! this is a stupid trick used in Kaggle
            # ! they flip the validation images, and then average the prediction over all the flips
            # ! we will set n_test=1 so that we do not flip.
            logits = torch.zeros((data.shape[0], args.num_ground_truth_label)).to(
                device
            )

            probs = torch.zeros((data.shape[0], args.num_ground_truth_label)).to(device)
            for I in range(n_test):
                l = model(get_trans(data, I), x_meta=meta)  # doing flips
                # if isinstance(l, tuple):
                #     l, _ = l
                logits += l
                probs += l.softmax(1)  # ! softmax if image has only 1 label.
                #probs += torch.sigmoid(l) # if image has multilabel 11/4

            # average over all the flips
            logits /= n_test
            probs /= n_test

            loss = criterion(logits, target)  # ! loss on the validation set.
            val_loss.append(loss.detach().cpu().numpy())

            logits_array.append(logits.detach().cpu())  # ! raw score output

            probs_array.append(probs.detach().cpu())  # ! probability output

            # ! original @target ground-truth, this allows for flexible accuracy metric
            ground_truth_array.append(target.detach().cpu())

            # img_path is a tuple of batch size. 
            img_path_array = img_path_array + list(img_path)

    # tally
    val_loss = np.mean(val_loss)
    # each logit is batch x num_label, we have many of these logits, so convert into 2D numpy
    logits_array = torch.cat(logits_array).numpy()
    probs_array = torch.cat(probs_array).numpy() # ! num_img x num_of_label 
    ground_truth_array = torch.cat(ground_truth_array).numpy() # ! num_img x num_of_ground_truth
    print("GROUNDTRUTH: ", ground_truth_array.shape)
    print("PROBARRAY: ", probs_array.shape)

    # ! global accuracy, each image has only 1 label in multi-class
    if probs_array.argmax(1).shape == ground_truth_array.shape:
        acc = (probs_array.argmax(1) == ground_truth_array).mean() * 100.0
        bal_acc = compute_balanced_accuracy_score(probs_array, ground_truth_array)
    else:
        acc = 0.5
        bal_acc = 0.5
        #changed this 11/4
        acc = (probs_array == ground_truth_array).mean() * 100.0
        print(probs_array)
        print(ground_truth_array)
        #print("ACC:", acc)
        #bal_acc = compute_balanced_accuracy_score(probs_array, ground_truth_array)

        
    # ! consider some imbalance when compute accuracy
    #bal_acc = compute_balanced_accuracy_score(probs_array, ground_truth_array)

    if also_get_output:
        return img_path_array, logits_array, probs_array, ground_truth_array, val_loss, acc, bal_acc
    else:
        # ! be careful here, we do multi-class vs multi-label ??
        return val_loss, acc, bal_acc


def make_or_load_model(args):
    # ---------------------------------------------------------------------------- #
    # define model

    model = ModelClass(
        model_arch_name=args.model_arch_name,
        num_ground_truth_label=args.num_ground_truth_label,
        n_meta_features=0,
        n_meta_dim=[512, 128],
        pretrained=True,
        dropout_rate_in_forward=args.dropout_rate_in_forward,
        dropout_rate_meta=0.3,
    )

    summary(model, input_size=(1, 3, args.image_size, args.image_size))
    
    model = model.to(device)

    # ! loading in a model
    if args.load_this_model is not None:
        print("\nloading {}\n".format(args.load_this_model))
        try:  # single GPU model_name
            if args.CUDA_VISIBLE_DEVICES is None:
                # ! ! setting strict=False
                model.load_state_dict(
                    torch.load(args.load_this_model, map_location=torch.device("cpu")),
                    strict=True,
                    map_location=torch.device("cpu"),
                )  # ! avoid error in loading model trained on GPU
            else:
                model.load_state_dict(torch.load(args.load_this_model, map_location=torch.device("cpu")), strict=True)
        except:  # multi GPU model_name
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(
                    args.load_this_model, map_location=torch.device("cpu")
                )
            else:
                state_dict = torch.load(args.load_this_model)
            state_dict = {
                k[7:] if k.startswith("module.") else k: state_dict[k]
                for k in state_dict.keys()
            }
            model.load_state_dict(state_dict, strict=True)

    # ! send to multiple gpus... only works well if we have model.forward, don't change forward func.
    if USE_MANY_GPU:
        # model = apex.parallel.convert_syncbn_model(model)
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus
        model = nn.DataParallel(model)

    return model


def train_one_fold(model, fold, df, transform_train_img, transform_val_img, criterion):
    # ---------------------------------------------------------------------------- #
    # use ground-truth panda df to create train and valid dataframe

    df_train = df[df["fold"] != fold]  # ! take out a fold and keep it as valid
    df_valid = df[df["fold"] == fold]  # ! take out a fold and keep it as valid

    # ---------------------------------------------------------------------------- #
    # define dataset and data loader

    train_dataset = DatasetFromPandaDf(
        df_train, "train", meta_features=None, transform=transform_train_img
    )

    valid_dataset = DatasetFromPandaDf(
        df_valid, "valid", meta_features=None, transform=transform_val_img
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=args.num_workers,
    )  # @RandomSampler shuffle the data

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )  # no need to shuffle data with @RandomSampler

    print(
        "train and valid data size {} , {}".format(
            len(train_dataset), len(valid_dataset)
        )
    )

    # ---------------------------------------------------------------------------- #
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)

    # ! https://discuss.pytorch.org/t/what-happend-if-i-dont-set-certain-parameters-with-requires-grad-false-but-exclude-them-in-optimizer-params/150374/2
    # optimizer = optim.Adam(
    #     [
    #         {"params": model.enet.parameters(), "lr": args.init_lr / 10},
    #         {"params": model.my_classifier.parameters()},
    #     ],
    #     lr=args.init_lr,
    # )

    if not args.no_scheduler:
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.n_epochs - 1
        )
        scheduler_warmup = GradualWarmupSchedulerV2(
            optimizer,
            multiplier=args.scheduler_scaler,
            total_epoch=1,
            after_scheduler=scheduler_cosine,
        )

    # ---------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------- #

    # define model names when saving
    model_name = os.path.join(
        args.model_dir, f"{args.run_name}_best_acc_fold{fold}.pth"
    )  # best model based on simple accuracy
    model_name_best_bal_acc = os.path.join(
        args.model_dir, f"{args.run_name}_best_bal_acc_fold{fold}.pth"
    )  # best based on balance accuracy
    model_name_last_epoch = os.path.join(
        args.model_dir, f"{args.run_name}_last_epoch_fold{fold}.pth"
    )
    model_name_best_loss = os.path.join(
        args.model_dir, f"{args.run_name}_best_val_loss_fold{fold}.pth"
    ) 

    # training the model

    best_epoch = 0  # ! early stop
    best_acc = 0.0
    best_bal_acc = 0.0
    best_val_loss = 2

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f"Fold {fold}, Epoch {epoch}")
        # scheduler_warmup.step(epoch - 1)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion=criterion, args=args
        )

        val_loss, acc, bal_acc = val_one_epoch(
            model, valid_loader, also_get_output=False, criterion=criterion, args=args
        )

        content = (
            time.ctime()
            + " "
            + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}'
        )
        print(content)  # print to log.out

        with open(
            os.path.join(args.log_dir, f"log_{args.run_name}.txt"), "a"
        ) as appender:
            appender.write(content + "\n")

        if not args.no_scheduler:
            scheduler_warmup.step()
            if epoch == 2:
                scheduler_warmup.step()  # bug workaround

        if acc > best_acc:  # ! save best model
            print(
                "best_acc ({:.6f} --> {:.6f}). Saving model ...".format(best_acc, acc)
            )
            torch.save(model.state_dict(), model_name)  #
            best_acc = acc

        if bal_acc > best_bal_acc:  # ! save model best when consider balance metric
            print(
                "best_bal_acc ({:.6f} --> {:.6f}). Saving model ...".format(
                    best_bal_acc, bal_acc
                )
            )
            torch.save(model.state_dict(), model_name_best_bal_acc)
            best_bal_acc = bal_acc
            best_epoch = epoch
        
        if val_loss < best_val_loss:
            print(
                "best_loss ({:.6f} --> {:.6f}). Saving model ...".format(best_val_loss, val_loss)
            )
            torch.save(model.state_dict(), model_name_best_loss)  #
            best_val_loss = val_loss

        # ! early stop based on acc. for our data
        if epoch - best_epoch > 20:
            break

    # ! end loop
    torch.save(model.state_dict(), model_name_last_epoch)  # why not?


def main():
    # ---------------------------------------------------------------------------- #
    # create model
    model = make_or_load_model(args)

    # ---------------------------------------------------------------------------- #
    # define loss function

    if args.weights_in_loss is not None:
        weight=[1.0, 1.0, 1.0] #, 3.0, 9.0] # ! [absent, present, fused]
    weight =[1.0, 1.0, 1.0] #, 3.0, 9.0]  # ! we can worry about weight of the labels later.
    class_weights = torch.FloatTensor(weight).cuda()

    # ! cross entropy when we do multi-label, otherwise, change to some other loss function
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.BCEWithLogitsLoss()
    # ---------------------------------------------------------------------------- #
    # define image augmentation transformation

    transform_train_img, transform_val_img, transform_resize_only = get_transforms(
        args.image_size
    )

    # ---------------------------------------------------------------------------- #
    # read in csv with ground-truth

    # img_path, ground_truth_label, fold
    # image_1, 1;0;1;0, 1 # you may need to do additional formatting
    # image_2, 0;1;1;0, 2
    # ! notice, csv contains label already coded as 1-hot.
    # ! image_1 has c2-c3-high-calicum, and c3-c4-high-calicum, then it gets [1,0,1,0]

    df = pd.read_csv(args.image_csv)  # ! need to format properly

    # ---------------------------------------------------------------------------- #
    # loop over each fold... only if we have 5-folds.

    if not args.run_on_test_img:
        # ! run 5 folds
        # ! should run each fold as its own submission job, so we can do 5-fold in parallel jobs.
        folds = [int(i) for i in args.fold.split(",")]
        print("\nfolds {}\n".format(folds))

        for fold in folds:
            train_one_fold(
                model, fold, df, transform_train_img, transform_val_img, criterion
            )

    elif args.run_on_test_img:
        # ---------------------------------------------------------------------------- #
        # ! note: validation step can be applied to test images.
        df = df#[0:10] # ? DEBUG
        test_dataset = DatasetFromPandaDf(
            df, "valid", meta_features=None, transform=transform_val_img
        )

        # no need to shuffle data with @RandomSampler
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

        img_path_array, logits_array, probs_array, ground_truth_array, val_loss, acc, bal_acc = (
            val_one_epoch(
                model, test_loader, also_get_output=True, criterion=criterion, args=args
            )
        )

        content = (
            time.ctime()
            + " "
            + f"test set, loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}"
        )
        print(content)  # print to log.out

        # ! write the predicted output into csv
        if args.csv_out_name is None: 
            args.csv_out_name = os.path.join(args.model_dir,'test_img_result.csv')
        util.WritePredictionToCsv(args.csv_out_name, img_path_array, probs_array, ground_truth_array)

        # can use @img_path_array, logits_array, probs_array
        # make a panda dataframe
        # @logits_array has size num_sample x num_ground_truth.

        # ---------------------------------------------------------------------------- #

        # ! add captum?
        # ! should do attribution on CPU, doesn't really need GPU at this point.
        if args.attribution_method is not None:
            # ! what if there are lot of images, and we want to parallelize?
            # ! take in range of images to be attributed.
            img_path_array = np.array(img_path_array)
            img_path_array = img_path_array[
                args.attribution_start_idx: args.attribution_end_idx
            ]

            probs_array = probs_array[
                args.attribution_start_idx: args.attribution_end_idx
            ]

            img_already_transformed = []
            image_resize_only = []
            for img in img_path_array:
                # https://github.com/albumentations-team/albumentations_examples/blob/main/notebooks/example.ipynb
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # KeyError: 'You have to pass data to augmentations as named arguments, for example: aug(image=image)'
                # need unsqueeze to turn 3 x 225 x 225 into 1 x 3 x 225 x 225.
                img_t = transform_val_img(image=img)['image'].astype(np.float32)
                img_t = img_t.transpose(2, 0, 1)
                img_t = torch.tensor(img_t).float().unsqueeze(0)
                img_already_transformed.append(img_t)
                #
                img_o = transform_resize_only(image=img)['image'].astype(np.float32)
                img_o = img_o.transpose(2, 0, 1)
                img_o = torch.tensor(img_o).float().unsqueeze(0)
                image_resize_only.append(img_o)

            # convert into batch-shape: batch x 3 x 225 x 225
            # https://discuss.pytorch.org/t/how-to-convert-a-list-of-tensors-to-a-pytorch-tensor/175666/2
            img_already_transformed = torch.cat(img_already_transformed, dim=0)
            image_resize_only = torch.cat(image_resize_only, dim=0)

            # ! run attribution

            device = torch.device("cpu")
            model.to(device)

            run_attribution = get_saliency.GetAttributionPlot(
                nn_model=model,  # resnet...
                img_already_transformed=img_already_transformed,  # already normalized/resized
                true_label_index=None,
                attribution_method=args.attribution_method,  # str: "Occlusion"...
                img_path=img_path_array,  # append original image name into attribution plot
                image_resize_only=image_resize_only,  # so we can plot the original image
                prediction=probs_array,  # if provided, use the "best" prediction , which can be wrong prediction
                output_dir=args.model_dir,  # save into the same fold of model that was used in attribution
            )
            run_attribution.make_saliency_plot(outlier_percent=10)

    else:
        pass


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    with open(os.path.join(args.log_dir, "commandline_args.txt"), "w") as f:
        json.dump(
            args.__dict__, f, indent=2
        )  # ! write out the command argument to backtrack and debug

    if "efficientnet" in args.model_arch_name:
        ModelClass = XrayClassifier
    else:
        raise NotImplementedError()

    USE_MANY_GPU = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed(seed=args.seed)  # ! set a seed, default to 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")

    main()
