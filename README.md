# AKU-progression-efficientnet

This is the github page for "Deep learning study of alkaptonuria spinal disease assesses global and regional severity and detects occult treatment status". 
Here, you will find the source code for the image segmentation processes and training of DL models. Additionally, you will find information about the nitisinone surveys as well as the full responses for each survey. Read below for more details about each directory.

-------------------------------------------------------------------------------------------------------------------------------
# AKU EfficientNet Models
In the directory `efficientnet` you will find the scripts to train DL models. Within `models`, you can find the trained model weights for narrowing, calcium, vacuum, global scores, and nitisinone. 

# Segmentation
In the directory `segmentation` you will find the links to the source code used to setup finetune-SAM and YOLO models. Additionally, you will find an example dataset (images and masks) used to finetune these models for cervical and lumbar X-ray images as well as example shell scripts and python training code specifically adapted from finetune-SAM and YOLO to fit our AKU dataset. More details about the function of each script is included in the README within `segmentation`. 

# Nitisinone Surveys
In the directory `surveys` you will find an example nitisinone status survey with real AKU cervical and lumbar images that was sent to participants (AKU Nitisinone Survey Example.pdf). You will also find a copy of all the responses, including the commentary that participants left about why they made certain predictions about a patient's nitisinone status (Nitisinone_Survey_Results.xlsx). Additionally, within this document, you can also find the participants self-rated confidence level about the survey.
