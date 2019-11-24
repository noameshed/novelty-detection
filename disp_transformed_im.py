# Display transformed images
from torchvision import models, transforms, datasets
import torch
from PIL import Image
import json
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.misc

# Transform for input images
transform = transforms.Compose([
    transforms.Resize(256),			# images should be 256x256
    transforms.CenterCrop(224),		# crop about the center to 224x224
    transforms.ToTensor(),			# convert to Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# Get all images in the CSV
path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Aves/'
savepath = 'C:/Users/noam_/Documents/Cornell/CS7999/11_25_19/alexnet_1000conf_more_data/aves_rf_images/actual_rel_pred_par/'

with open('C:/Users/noam_/Documents/Cornell/CS7999/11_25_19/alexnet_1000conf_more_data/aves_rf_results.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        imname = row[0]
        actual = row[1]
        pred = row[2]

        if actual == 'relative in imagenet' and pred == 'parent in imagenet':
            # Find the image in the folder path
            for folder in os.listdir(path):
                subpath = path + folder + '/'
                if imname in os.listdir(subpath):
                    impath = subpath + imname
                    image = Image.open(impath)
                    
                    img_t = transform(image).permute(1,2,0)
                    fig = plt.imshow(img_t)
                    plt.axis('off')
                    # plt.savefig(savepath + folder + ' ' + imname, bbox_inches='tight', pad_inches=0)
                    scipy.misc.imsave(savepath + folder + ' ' + imname, img_t)
                    # plt.show()