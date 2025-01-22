import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
import re, pickle
from matplotlib.colors import LinearSegmentedColormap

# import torchvision
# from torchvision import models
# from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from tqdm import tqdm

class GetAttributionPlot:
    def __init__(
        self,
        nn_model,  # resnet...
        img_already_transformed,  # already normalized/resized
        true_label_index,
        attribution_method,  # str: "Occlusion"...
        img_path,  # append original image name into attribution plot
        image_resize_only,  # so we can plot the original image
        prediction,  # if provided, use the "best" prediction , which can be wrong prediction
        output_dir,
    ):
        self.nn_model = nn_model
        # @ img_already_transformed can be a batch x 3 x 225 x 225
        self.img_already_transformed = img_already_transformed
        self.true_label_index = true_label_index
        self.attribution_method = attribution_method
        self.img_path = img_path
        self.image_resize_only = image_resize_only

        if (true_label_index is None) and (prediction is not None):
            self.label_idx_in_saliency = torch.tensor(np.argmax(prediction, 1))  # take top prediction
            print ('self.label_idx_in_saliency',self.label_idx_in_saliency)
        else:
            self.label_idx_in_saliency = true_label_index

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def set_and_run_saliency_method(self, index):
        # ! https://captum.ai/tutorials/TorchVision_Interpret
        # @self.this_attribution is a tensor format. batch x channel x height x width
        image = self.img_already_transformed[index].unsqueeze(0)
        label = self.label_idx_in_saliency[index]

        print ('img name and shape', self.img_path[index], image.shape)

        if self.attribution_method == "Occlusion":
            self.this_saliency_call = Occlusion(self.nn_model)
            this_attribution = self.this_saliency_call.attribute(
                image,
                strides=(3, 10, 10),
                target=label,
                sliding_window_shapes=(3, 20, 20),
                baselines=0,
            )
        if self.attribution_method == "GradientShap":
            self.this_saliency_call = GradientShap(self.nn_model)
            rand_img_dist = torch.cat(
                [image * 0, image * 1]
            )
            this_attribution = self.this_saliency_call.attribute(
                image,
                n_samples=50,
                stdevs=0.0001,
                baselines=rand_img_dist,
                target=label,
            )
        # @this_attribution maybe in GPU, send back to CPU, and do plotting (which does not need GPU anyway)
        this_attribution = (
            this_attribution.cpu().detach().numpy()
        )  # ! batch x c x h x w ?
        return this_attribution

    def make_saliency_plot(self, outlier_percent):
        #
        # set cmap=default_cmap_black_white inside viz.visualize_image_attr
        # can be useful if we do "overlapping"
        # default_cmap_black_white = LinearSegmentedColormap.from_list(
        #     "custom black",  # white-->black color gradient scale
        #     [(0, "#ffffff"), (1, "#000000")],
        #     N=256,
        # )

        # we can only plot each image one at a time ?
        # well... we can use visualize_image_attr_multiple, but how do we plot each figure separately?
        for b in tqdm(range(self.img_already_transformed.shape[0])):

            # ! run saliency, can be done in batch-mode
            this_attribution = self.set_and_run_saliency_method(b)

            # convert to numpy
            attribution_np = this_attribution.squeeze()
            # ! (224, 224, 3), h x w x chanels, allows us to plot with matplotlib
            attribution_np = np.transpose(attribution_np, (1, 2, 0))

            # convert original image (resized only) into numpy to plot too
            original_img = self.image_resize_only[b].squeeze().detach().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))

            # ! heatmap overlap on image @blended_heat_map
            output_plot = viz.visualize_image_attr(
                attribution_np,  # ! (224, 224, 3), h x w x chanels
                original_img,
                method="blended_heat_map",
                show_colorbar=False,
                sign="positive",
                outlier_perc=outlier_percent,
            )

            # save the plot
            temp = re.sub(
                r"\.(jpg|png|jpeg)", "_blend_heatmap_posi.png", self.img_path[b]
            ).split("/")[-1] # append '_blend_heatmap_posi' to original name of image 
            # return 2 items, https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py#L462
            # hence, we use output_plot[0]
            output_plot[0].savefig(
                os.path.join(self.output_dir, temp),
                bbox_inches="tight",
                pad_inches=0.0,
                dpi=400,
            )

            # ! heatmap alone which can be useful take take "average"
            # ! "average" would imply input images have to be "uniformly aligned"
            output_plot = viz.visualize_image_attr(
                attribution_np,
                original_img,
                method="heat_map",
                show_colorbar=False,
                sign="positive",
                outlier_perc=outlier_percent,
            )

            temp = re.sub(
                r"\.(jpg|png|jpeg)", "_heatmap_posi.png", self.img_path[b]
            ).split("/")[-1]
            output_plot[0].savefig(
                os.path.join(self.output_dir, temp),
                bbox_inches="tight",
                pad_inches=0.0,
                dpi=400,
            )
