#!/usr/bin/env python
"""
Psychopy experiment for image selection
"""
from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
import numpy, random, os, ctypes

class BirdSimExp():

    def __init__(self):
        self.name='Bird Similarity Experiment'
        self.expInfo = self.getSessionInfo()

        # Initialize path for bird images
        self.path = os.getcwd()
        self.impath = self.path + '/images/'
        self.all_birds = os.listdir(self.impath)

        # make a text file to save data
        fileName = self.expInfo['ID'] + self.expInfo['dateStr']
        self.dataFile = open(self.path+'/data/'+fileName+'.csv', 'w+')  # a simple text file with comma-separated-values
        self.dataFile.write('leftIm,rightIm,userChoice,responseTime\n')

        # create window
        user32 = ctypes.windll.user32
        res = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.win = visual.Window([res[0], res[1]], allowGUI=True, color=[0.2,0.2,0.2],
                monitor='testMonitor', units='height')

        # Define the rating scale parameters
        self.scaleMsg = visual.TextStim(self.win, pos=[0.5, 0.3], height=0.02, alignHoriz='center',
                text="Rate how similar the two images below are on a scale from 1-7. Then press Enter.")
        self.rating = visual.RatingScale(self.win,pos=[0, -.4], low=1, high=7, respKeys=['1','2','3','4','5','6','7'],
                acceptKeys=['return', 'space'], showAccept=False, textColor=[0.2,0.2,0.2] )
        self.labels = visual.TextStim(self.win, pos=[0.72,-.22], height=0.02, alignHoriz='center',
                text="1           2          3          4          5          6          7")
        self.diff_label = visual.TextStim(self.win, pos=[0.5, -.22], height=.02, text='very different')
        self.sim_label = visual.TextStim(self.win, pos=[1.35, -.22], height=.02, text='very similar')

    def quit(self):
        # Close the data file
        self.dataFile.close()

        # Clear the screen
        # TODO: Set a 'thank you' message
        self.win.flip()
        event.waitKeys()  

        # Close the window and end processes
        self.win.close()
        core.quit()

    def getSessionInfo(self):
        try:  # try to get a previous parameters file
            expInfo = fromFile(os.getcwd() + '/lastParams.pickle')
        except:  # if not there then use a default set
            expInfo = {'ID':'ne236'}
        expInfo['dateStr'] = data.getDateStr()  # add the current time

        # present a dialogue to change params
        dlg = gui.DlgFromDict(expInfo, title='Image Similarity', fixed=['dateStr'])
        if dlg.OK:
            toFile(os.getcwd() + 'lastParams.pickle', expInfo)  # save params to file for next time
        else:
            core.quit()  # the user hit cancel so exit

        return expInfo

    def instructions(self):
        
        # display instructions and wait
        msg1 = visual.TextStim(self.win, pos=[0.5, 0.1], height = 0.03,
            text="In this experiment, you will be shown pairs of images. \n\nChoose a value (1-7) on the scale to describe how similar they are.\n\n1 is least similar and 7 is most similar. \n\nUse the 1-7 keyboard keys to select the value. Then press Enter.")

        msg2 = visual.TextStim(self.win, pos=[0.5, -0.2], height=0.03, text='Press any key when you are ready to begin.')

        msg1.draw()
        msg2.draw()

        self.win.flip()  #to show the messages
        #pause until there's a keypress
        event.waitKeys()

    def practiceRound(self, n):
        # Run a few practice rounds for calibration
        msg1 = visual.TextStim(self.win, pos=[.5, 0.1], height=0.03,
            text="You will now do a practice round. You will see pairs of images and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")

        msg1.draw()
        self.win.flip()
        event.waitKeys()

        for i in range(n):
            # choose 2 random images of 2 types of birds
            [A, B] = random.sample(self.all_birds, 2)
            choice1 = random.choice(os.listdir(self.impath+A))
            choice2 = random.choice(os.listdir(self.impath+B))

            leftIm = visual.ImageStim(self.win, image=self.impath+A+'/'+choice1, 
                flipHoriz=True, pos=(-10,3), units='deg', size=(10,10))

            rightIm = visual.ImageStim(self.win, image=self.impath+B+'/'+choice2, 
                flipHoriz=True, pos=(10,3), units='deg', size=(10,10))    
            
            self.rating.reset()

            # Wait for participant response
            while self.rating.noResponse:
                self.drawAll(leftIm, rightIm, self.scaleMsg, self.rating, 
                    self.labels, self.diff_label, self.sim_label, self.win)

                # Close the window if they hit 'escape'
                if event.getKeys(['escape']):
                    core.quit()

    def recordedTrials(self, n):
        # Show messages telling the participant they are about to begin the real experiment
        msg1 = visual.TextStim(self.win, pos=[.5, 0.1], height=0.03,
            text="You will now begin the experiment. You will see pairs of images and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")

        msg1.draw()
        self.win.flip()
        event.waitKeys()
        timer = core.Clock()
        t0=0
        t1=0

        for i in range(n): # 200 birds is about 10 minutes
            # choose 3 random images of 3 types of birds
            [A, B] = random.sample(self.all_birds, 2)
            choice1 = random.choice(os.listdir(self.impath+A))
            choice2 = random.choice(os.listdir(self.impath+B))

            # Create the two image objects
            leftIm = visual.ImageStim(self.win, image=self.impath+A+'/'+choice1, 
                flipHoriz=True, pos=(-10,3), units='deg', size=(10,10))

            rightIm = visual.ImageStim(self.win, image=self.impath+B+'/'+choice2, 
                flipHoriz=True, pos=(10,3), units='deg', size=(10,10))

            self.rating.reset()
            t0 = timer.getTime()
            # Wait for participant response
            while self.rating.noResponse:
                self.drawAll(leftIm, rightIm, self.scaleMsg, self.rating, 
                    self.labels, self.diff_label, self.sim_label, self.win)

                # Close the window if they hit 'escape'
                if event.getKeys(['escape']):
                    core.quit()
                    
            t1 = timer.getTime()
            self.dataFile.write('%s,%s,%s,%s\n' %(A+'/'+choice1, B+'/'+choice2, self.rating.getRating(), t1-t0))

    def drawAll(self, *argv):
        for arg in argv:
            arg.draw()

if __name__ == '__main__': 
    b = BirdSimExp()
    b.instructions()
    b.practiceRound(15)
    b.recordedTrials(200)
    b.quit()

    
    # Survey user's familiarity with birds
    """
    msg1 = visual.TextStim(win, pos=[.5, 0.1], height=0.03,
        text="You will now complete a short survey to complete the experiment. Press any key when you are ready to begin.")

    msg1.draw()
    win.flip()
    event.waitKeys()

    ques1 = visual.TextStim(win, pos=[.5, 0.1], height=0.03,
        text="How would you rate your familiarity with birds?")
    ratingScale=visual.RatingScale(win, choices=['I know nothing about birds','I am an active birdwatcher/researcher'],
        markerStart=0.5, singleClick=True)

    ques1.draw()
    ratingScale.draw()
    win.flip()
    event.waitKeys()
    """
    