import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import shutil

def getMinAndMax(datapath):
    # Get the min and max distance scores from *all* of the pairs of image distances
    maxdist = 0
    mindist = np.inf
    # Loop through all files with scores
    for f in tqdm(os.listdir(datapath)):
        # Open as pandas DF
        df = pd.read_csv(datapath+f)
        score = df['score']
        curmin = min(score)
        curmax = max(score)

        # Compute overall min and max
        if curmin < mindist:
            mindist = curmin
        if curmax > maxdist:
            maxdist = curmax
        
        # clear data from dataframe
        del df       

    return mindist, maxdist

def normalize(datapath):
    # Get normalized (0-1) distance scores and save them in files
    print('Normalizing scores')
    # Compute overall min and max scores
    mindist, maxdist = getMinAndMax(datapath)
    
    for f in tqdm(os.listdir(datapath)):
        # Open file as pandas df
        df = pd.read_csv(datapath+f)
        score = df['score']

        # Adjust score to be between 0 and 1
        normScores = (score - mindist)/(maxdist-mindist)
        df['normScores'] = normScores

        # Overwrite CSV with new normalized scores
        df.to_csv(datapath+f)
        # Clear memory from dataframe
        del df
        
def stratifyPairs(datapath, savepath, n):
    """
    Create n lists of image pairs with increasing distance scores
    such that list 1 has image pairs with scores 0 to 1/n,
    list 2 has image pairs with scores 1/n to 2/n, etc.
    """ 
    print('Stratifying image pairs into score bins')
    # Create score limits for each set of image pairs
    split = np.linspace(0,1, n+1)
    split[-1] += 0.001

    # Go through each document and split/store the image pairs
    for f in tqdm(os.listdir(datapath)):
        # Open file as pandas df
        df = pd.read_csv(datapath+f)
        im1 = np.asarray(df['image1'])
        im2 = np.asarray(df['image2'])
        score = np.asarray(df['normScores'])

        # Initialize indices array
        allScores = []
        for i in range(len(split)-1):
            df = pd.DataFrame(columns=['image1', 'image2','normScores'])
            allScores.append([])

        # Go through the different split options:
        for i in range(1, len(split)):
            idx = np.argwhere((score >= split[i-1]) & (score < split[i]))
            print(i, len(idx))
            # Add pair info to the appropriate place in the list
            for j in idx:
                triple = [im1[j].item(), im2[j].item(), score[j].item()]
                allScores[i-1].append(triple)

        # Save the arrays as CSV docs
        for i in range(len(allScores)):
            left = str(round(split[i], 2))
            right = str(round(split[i+1], 2))
            name =  str(i)+'_scores_'+left+'_'+right+'.csv'
            df = pd.DataFrame(allScores[i])
            df.to_csv(savepath+name, mode='a', header=False, index=False)

def selectSubset(path, pairs):
    # Randomly select a subset of image pairs from each of the stratified bins
    binfiles = os.listdir(path)[::-1]
    for f in tqdm(binfiles):

        # Open file and store contents
        df = pd.read_csv(path+f)

        print(df.head)
        # Randomly select pairs number of image pairs
        idx = np.random.choice(len(df), pairs)
        df = df.iloc[idx,:]

        # Save the selected data with a new name
        savename = 'subset_'+f
        df.to_csv(path + savename, header=False, index=False)
        del df

def saveUsedImages(namepath, imgpath, savepath):
    # Takes all of the images stored in the path files and copies them into a new directory
    binfiles = os.listdir(namepath)[::-1]
    for f in tqdm(binfiles):

        # Open file and store contents
        df = pd.read_csv(namepath+f)

        for index, row in df.iterrows():
            # Get the paths for the two birds in the pair
            path1 = row[0]
            path2 = row[1]

            # Extract the species name and image name
            species1 = path1.split('/')[0].strip()
            species2 = path2.split('/')[0].strip()
            bird1 = path1.split('/')[1].split('_')[0].strip() + '.jpg'
            bird2 = path2.split('/')[1].split('_')[0].strip() + '.jpg'

            # Try to create a new image folder for each species
            if not os.path.exists(savepath+species1):
                os.makedirs(savepath+species1)
            if not os.path.exists(savepath+species2):
                os.makedirs(savepath+species2)

            # Copy the images into their new folders if not there already
            savefile1 = savepath + species1 + '/' + bird1
            savefile2 = savepath + species2 + '/' + bird2
            if not os.path.exists(savefile1):
                shutil.copy(imgpath + species1 + '/' + bird1, savefile1)
            if not os.path.exists(savefile2):
                shutil.copy(imgpath + species2 + '/' + bird2, savefile2)


if __name__ == "__main__":
    datapath = os.getcwd() + '/image_distances/'
    pairscorespath = os.getcwd() + '/stratified_img_pairs/'
    imgpath = os.getcwd() + '/images/Aves/'
    imgsubsetpath = os.getcwd() + '/images/Aves_sub/'

    #normalize(datapath)        # Set scores to be between 0 and 1
    #stratifyPairs(datapath, savepath, 7)       # Split scores by CNN score into 7 bins

    # There are too many image pairs and the data files are too large
    # Precompute the image pairs that we want users to see
    pairs = 150   # How many image pairs are chosen from each set
    # 100 images -> 18% overlap between participants
    # 150 images -> 8% overlap between participants
    # 200 images -> 4.6% overlap between participants
    #selectSubset(savepath, pairs)

    # Copy the randomly chosen images to a new folder
    saveUsedImages(pairscorespath, imgpath, imgsubsetpath)

