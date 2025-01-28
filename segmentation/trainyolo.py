import os
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import json 
import skimage as ski
from glob import glob
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['YOLO_CONFIG_DIR']= '/data/flahartyka/'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

save_dir = '/data/flahartyka/yolo_cervical/training' # Save everything here

# Load a model
model = YOLO("yolov8n.pt")  # load pretrained YOLO model (recommended for training)

# Train the model
model.train(data="/data/flahartyka/yolo_cervical/training/dataset.yaml", epochs=50)

#metrics = model.val()  # evaluate model performance on the validation set
#results = model("/data/flahartyka/yolo/training/AKU102_461_lumbar.png")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

# Load the exported ONNX model
onnx_model = YOLO("/data/flahartyka/yolo_cervical/runs/detect/train3/weights/best.onnx")

# Run inference
results = onnx_model("/data/flahartyka/Multilabel/Left_Cervical") #predict on images
os.mkdir('AKU_results')

for result in results:
    img_name = result.path.split('/')[-1]
    file = '/data/flahartyka/yolo_cervical/AKU_results/'
    filename = file + img_name
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=filename)  # save to disk

