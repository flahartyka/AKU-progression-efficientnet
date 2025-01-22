import albumentations
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetFromPandaDf(Dataset):
    def __init__(
        self,
        panda_df,
        mode,
        meta_features=None,
        transform=None,
    ):
        # read in csv with ground-truth
        self.panda_df = panda_df.reset_index(drop=True)
        # img_path, ground-truth-label, fold, age, gender
        # image_1, 1;0;1;0, 1 # you may need to do additional formatting
        # image_2, 0;1;1;0, 2
        # ! notice, csv contains label already coded as 1-hot.
        # ! image_1 has c2-c3-high-calicum, and c3-c4-high-calicum, then it gets [1,0,1,0]

        #self.mode = mode
        #self.mode = "soft_label"
        self.mode = "multilabel" #Change the mode here

        self.meta_features = meta_features  # ! later, we can add in "age/gender..."

        self.transform = transform

    def __len__(self):
        return self.panda_df.shape[0]

    def __getitem__(self, index):
        row = self.panda_df.iloc[index]  # get a row in the csv

        img_path = row["img_path"]
        image = cv2.imread(img_path)  # ! read a file path from @csv,
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ! transform image in pytorch tensor data type to be used with the model
        res = self.transform(image=image)  # ! uses albumentations
        image = res["image"].astype(np.float32)
        # makes channel x h x w instead of h x w x c
        image = image.transpose(2, 0, 1)

        if self.meta_features is not None:  # ! age and gender?
            data = (
                torch.tensor(image).float(),
                torch.tensor(row[self.meta_features]).float(),
            )
        else:
            # return image as pytorch tensor format.
            data = torch.tensor(image).float()

        # during testing phase have no ground-truth
        if self.mode == "test_with_no_label":
            return data, img_path  # ! return image path so we can debug easier
        elif self.mode == "soft_label":
            label_np = np.fromstring(row["ground_truth_label"], dtype=float, sep=";") # ! works with soft label
            return data, torch.FloatTensor(label_np), img_path
        elif self.mode == "multilabel":
            label_np = np.fromstring(row["ground_truth_label"], dtype=float, sep=";") # ! works with soft label
            return data, torch.tensor(label_np).float(), img_path            
        else:  # during training and evaluation, we have ground-truth
            # ! notice using sep=';' on the panda string
            label_np = np.fromstring(row["ground_truth_label"], dtype=float, sep=";") # ! works with soft label
            label_np = label_np.argmax() # ! take 1 single highest index as ground-truth
            return data, torch.tensor(label_np).long(), img_path


def get_transforms(image_size):
    transform_train_img = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(
                p=0.5
            ),  # ! able to change the probability of being flip. p=1 --> always flip
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2),p=0.5),
            # albumentations.RandomBrightness(limit=0.2, p=0.5),
            # albumentations.RandomContrast(limit=0.2, p=0.5),
            albumentations.OneOf(
                [
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 10.0)),
                ],
                p=0.25,
            ),
            # albumentations.OneOf([
            #     albumentations.OpticalDistortion(distort_limit=1.0),
            #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            #     albumentations.ElasticTransform(alpha=3),
            # ], p=0.7),
            albumentations.CLAHE(
                clip_limit=4.0, p=0.5
            ),  # ! https://github.com/albumentations-team/albumentations?tab=readme-ov-file#albumentations
            albumentations.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=45, border_mode=0, p=0.5
            ),
            albumentations.Resize(image_size, image_size),
            # albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
            albumentations.Normalize(),
        ]
    )

    # ! just resize, and convert 0-255 -> 0-1 scale.
    transform_val_img = albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
        ]
    )

    # ! only resize image, this can be useful for viewing original image
    transform_resize_only = albumentations.Compose(
        [albumentations.Resize(image_size, image_size)]
    )

    return transform_train_img, transform_val_img, transform_resize_only
