import numpy as np
import os
import pandas as pd
from tqdm import tqdm

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


if __name__ == "__main__":
    datapath = os.getcwd() + '/image_distances/'
    savepath = os.getcwd() + '/stratified_img_pairs/'
    normalize(datapath)
    stratifyPairs(datapath, savepath, 7)

