#!/usr/bin/env python
"""
Psychopy experiment for image selection
"""
from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
import numpy, random, os, ctypes, csv

class BirdSimExp():

    def __init__(self, prompt_type):
        self.name='Bird Similarity Experiment'
        self.expInfo = self.getSessionInfo()
        if prompt_type == 0:
            self.prompt = 'images'
        else:
            self.prompt = 'birds'

        # Initialize paths for bird images and scores
        self.path = os.getcwd()
        self.impath = self.path + '/images/Aves/'
        self.scorepath = self.path + '/stratified_img_pairs/'
        self.all_birds = os.listdir(self.impath)
        self.pairScores = os.listdir(self.scorepath)

        # Open data files
        self.data = []
        for i in range(len(os.listdir(self.scorepath))):
            self.data.append([])

        for i, f in enumerate(os.listdir(self.scorepath)):
            with open(self.scorepath + f, 'r') as fopen:
                for line in fopen:
                    self.data[i].append(line)
            
        # Make a new csv file to save data
        fileName = self.expInfo['Participant ID'] + '_PT=' + self.prompt + '_' + self.expInfo['dateStr']
        self.dataFile = open(self.path+'/data/'+fileName+'.csv', 'w+')  # a simple text file with comma-separated-values
        self.dataFile.write('leftIm,rightIm,userChoice,cnnRating,responseTime\n')

        # Create a window using the monitor's dimensions
        user32 = ctypes.windll.user32
        res = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)    # Get monitor dimensions
        self.win = visual.Window([res[0], res[1]], allowGUI=True, color=[0.2,0.2,0.2],
                monitor='testMonitor', units='cm', fullscr=True)

        # Define the rating scale parameters
        self.scaleMsg = visual.TextStim(self.win, pos=[0,10], height=1, wrapWidth=40,           #[0.5, 0.3]
                text="Rate how similar the two " + self.prompt + " below are on a scale from 1-7. Then press Enter.")
        self.rating = visual.RatingScale(self.win,pos=[0,-.3], low=1, high=7, respKeys=['1','2','3','4','5','6','7'],      #[0, -.4]
                acceptKeys=['return', 'space'], showAccept=False, textColor=[0.2,0.2,0.2] )
        self.labels = visual.TextStim(self.win, pos=[0,-6], height=.8, wrapWidth=40,            # [0.72,-.22], 0.02
                text="1           2          3          4          5          6          7")
        self.diff_label = visual.TextStim(self.win, pos=[-14,-6], height=.8, text='very different')        # [0.5, -.22]
        self.sim_label = visual.TextStim(self.win, pos=[14,-6], height=.8, text='very similar')         #[1.35, -.22]

    def drawAll(self, *argv):
        for arg in argv:
            arg.draw()

    def getSessionInfo(self):
        expInfo = {'Participant ID':'', 'Experimenter ID':''}
        expInfo['dateStr'] = data.getDateStr()  # add the current time

        # present a dialogue to change params
        dlg = gui.DlgFromDict(expInfo, title='Image Similarity', fixed=['dateStr'])
        if not dlg.OK:
            core.quit()  # the user hit cancel so exit

        return expInfo

    def instructions(self):
        # display instructions and wait
        msg1 = visual.TextStim(self.win, pos=[0, 0], height=1, wrapWidth=40,                 # [0.5, 0.1]    0.03
            text=
            "In this experiment, you will be shown pairs of bird photos.\n\n\
You will either be asked to rate the similarity of the images themselves, or to rate the similarity of the birds in the images.\n\n\
Press any key when you are ready to begin.")

        msg2 = visual.TextStim(self.win, pos=[0, 0], height=1, wrapWidth=40,                 # [0.5, 0.1]    0.03
            text=
            "Please rate the similarity between the pairs of" + self.prompt + ".\n\n\
Choose a value (1-7) on the scale to describe how similar they are.\n\n\
1 is least similar and 7 is most similar. Use the 1-7 keyboard keys to select the value.\n\n\
Then press Enter. Press any key when you are ready to begin.")

        msg1.draw()
        self.win.flip()     # to show the messages
        event.waitKeys()    # pause until there's a keypress

        msg2.draw()
        self.win.flip()
        event.waitKeys()

    def getBirdPair(self):
        # Select a pair of birds to show the participant

        # Randomly select a difference bin to choose from
        idx = numpy.random.choice(range(len(self.data)))
        d = self.data[idx]

        # Randomly select a pair of birds
        idx = numpy.random.choice(len(d))
        row = d[idx].split(',')
        A = row[0].split('_')[0]
        B = row[1].split('_')[0]
        rating = row[2].strip()

        # Get the paths to the bird images
        #[A, B] = random.sample(self.all_birds, 2)
        path1 = A + '.jpg'
        path2 = B + '.jpg'
            
        # Randomize the order in which the images are shown
        impaths = [path1, path2]
        random.shuffle(impaths)
        return impaths[0], impaths[1], rating

    def trials(self, n, writeData, practice):
        # Show message telling the participant they are starting a warmup trial
        msg = visual.TextStim(self.win, pos=[0,0], height=1, wrapWidth=40,
                text="You will now begin a practice round. You will see pairs of " + self.prompt + " and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")
        if not practice:   # Show messages telling the participant they are about to begin the real experiment
            msg = visual.TextStim(self.win, pos=[0,0], height=1, wrapWidth=40,    
                text="You will now begin the experiment. You will see pairs of " + self.prompt + " and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")
            
        msg.draw()
        self.win.flip()
        event.waitKeys()

        # Initialize the response timer 
        timer = core.Clock()

        for i in range(n): # 100 birds is about 10 minutes
            # choose an image pair from the precomputed list of image pairs
            [path1, path2, cnn_rating] = self.getBirdPair()

            # Create the two image objects
            leftIm = visual.ImageStim(self.win, image=self.impath + path1, 
                flipHoriz=True, pos=(-10,3), units='deg', size=(10,10))
            rightIm = visual.ImageStim(self.win, image=self.impath + path2, 
                flipHoriz=True, pos=(10,3), units='deg', size=(10,10))

            self.rating.reset()
            t0 = timer.getTime()
            # Display images and wait for participant response
            while self.rating.noResponse:
                self.drawAll(leftIm, rightIm, self.scaleMsg, self.rating, 
                    self.labels, self.diff_label, self.sim_label)
                self.win.flip()

                # Close the window if they hit 'escape'
                if event.getKeys(['escape']):
                    core.quit()
                    
            t1 = timer.getTime()
            if writeData:
                self.dataFile.write('%s,%s,%s,%s,%s\n' %(path1, path2, self.rating.getRating(), str(cnn_rating), t1-t0))

    def thankyou(self):
        # display end-of-experiment message
        msg = visual.TextStim(self.win, pos=[0, 0], height=1, wrapWidth=40,                 # [0.5, 0.1]    0.03
            text=
            "You have now completed the experiment. Thank you for participating. Press any key to exit.")
        msg.draw()

        self.win.flip()     # to show the messages
        event.waitKeys()    # pause until there's a keypress
        self.quit()

    def quit(self):
        # Close the data files
        self.dataFile.close()
        self.surveyFile.close()

        # Clear the screen
        self.win.flip()

        # Close the window and end processes
        self.win.close()
        core.quit()

if __name__ == '__main__': 

    # Randomly select the participant's prompt type ('bird' or 'image')
    prompt_type=0
    if numpy.random.rand() < 0.5:
        prompt_type=1

    b = BirdSimExp(prompt_type)
    b.instructions()
    b.trials(15, True, True)     # Run warmup trials (recorded)
    b.trials(300, True, False)     # Run experiment trials (recorded)
    b.thankyou()