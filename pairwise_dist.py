"""Compute pairwise distances between all images in directory"""

import csv
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def process_im(im_path):
    pil_im =  Image.open(im_path).convert('RGB')
    pil_im=pil_im.resize((224, 224))
    im_as_arr = np.float32(pil_im)
    pil_im.close()
    return im_as_arr

def euclid_dist(im1, im2):
    return np.sum((im1-im2)**2)

def kl(im1, im2):
    eps = 0.00001
    im1 = im1 + eps
    im2 = im2 + eps
    return np.sum(np.where(im1 != 0, im1*np.log(im1/im2), 0))

if __name__ == "__main__":
    
    path = 'D:/noam_/Cornell/CS7999/iNaturalist/gradients/Aves_layer3/'
    savepath='D:/noam_/Cornell/CS7999/iNaturalist/layer3_dists/'
    # Get a list of paths to every single bird image
    allbirds = []
    for s in os.listdir(path):
        speciespath = path + s + '/'
        for im in os.listdir(speciespath):
            fullpath = speciespath + im
            allbirds.append(fullpath)

    print('birds in allbirds:', len(allbirds))

    # Create pandas dataframe to save data
    data = [['image1', 'image2', 'scoretype', 'score', 'scoretype', 'score']]

    last_species=allbirds[0].split('/')[-2]
    counter=1
    for i in tqdm(range(len(allbirds)-1)):
        bird1 = allbirds[i].split('Aves_layer3/')[1]
        cur_species = bird1.split('/')[0]
        im1 = process_im(allbirds[i])
        
        # Save file whenever we switch to a new species for i (bird1)
        if last_species != cur_species:
            
            last_species = cur_species
            name = 'pairwise_dists'+str(counter)+'.csv'
            counter+=1
            with open(savepath+name, 'w') as f:
                wr = csv.writer(f, lineterminator='\n')
                wr.writerows(data)
                del data
                data =  [['image1', 'image2', 'scoretype', 'score', 'scoretype', 'score']]

        # Compare the current bird to every other bird in our database
        #for j in range(0,i):
        for j in range(i+1, len(allbirds)):
            im2 = process_im(allbirds[j])
            disteuc = euclid_dist(im1, im2)
            distkl = kl(im1, im2)

            bird2 = allbirds[j].split('Aves_layer3/')[1]
            row = [bird1, bird2, 'euclid', disteuc, 'kl', distkl]
            data.append(row)
        
