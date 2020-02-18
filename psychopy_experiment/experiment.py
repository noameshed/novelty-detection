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

        # Make a text file to save data
        fileName = self.expInfo['ID'] + self.expInfo['dateStr']
        self.dataFile = open(self.path+'/data/'+fileName+'.csv', 'w+')  # a simple text file with comma-separated-values
        self.dataFile.write('leftIm,rightIm,userChoice,responseTime\n')

        # Create window
        user32 = ctypes.windll.user32
        res = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)    # Get monitor dimensions
        self.win = visual.Window([res[0], res[1]], allowGUI=True, color=[0.2,0.2,0.2],
                monitor='testMonitor', units='cm', fullscr=True)

        # Define the rating scale parameters
        self.scaleMsg = visual.TextStim(self.win, pos=[0,10], height=1, alignHoriz='center', wrapWidth=40,           #[0.5, 0.3]
                text="Rate how similar the two images below are on a scale from 1-7. Then press Enter.")
        self.rating = visual.RatingScale(self.win,pos=[0,-.3], low=1, high=7, respKeys=['1','2','3','4','5','6','7'],      #[0, -.4]
                acceptKeys=['return', 'space'], showAccept=False, textColor=[0.2,0.2,0.2] )
        self.labels = visual.TextStim(self.win, pos=[0,-6], height=.8, alignHoriz='center', wrapWidth=40,            # [0.72,-.22], 0.02
                text="1           2          3          4          5          6          7")
        self.diff_label = visual.TextStim(self.win, pos=[-14,-6], height=.8, text='very different')        # [0.5, -.22]
        self.sim_label = visual.TextStim(self.win, pos=[14,-6], height=.8, text='very similar')         #[1.35, -.22]

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
        h = 1
        w = 40
        msg1 = visual.TextStim(self.win, pos=[0, 0], height=h, wrapWidth=w,                 # [0.5, 0.1]    0.03
            text=
            "In this experiment, you will be shown pairs of images.\n\n\
Choose a value (1-7) on the scale to describe how similar they are.\n\n\
1 is least similar and 7 is most similar. Use the 1-7 keyboard keys to select the value.\n\n\
Then press Enter. Press any key when you are ready to begin.")

        msg2 = visual.TextStim(self.win, pos=[0, 3], height=h, wrapWidth=w,
            text="Choose a value (1-7) on the scale to describe how similar they are.")
        msg3 = visual.TextStim(self.win, pos=[0, 1], height=h, wrapWidth=w, 
            text="1 is least similar and 7 is most similar.")
        msg4 = visual.TextStim(self.win, pos=[0, -1], height=h, wrapWidth=w, 
            text="Use the 1-7 keyboard keys to select the value. Then press Enter.")
        msg5 = visual.TextStim(self.win, pos=[0, -3], height=h, wrapWidth=w, 
            text='Press any key when you are ready to begin.')
        
        msg1.draw()
        #self.drawAll(msg1, msg2, msg3, msg4, msg5)

        self.win.flip()     # to show the messages
        event.waitKeys()    # pause until there's a keypress

    def trials(self, n, writeData):
        # Show messages telling the participant they are about to begin the real experiment
        msg1 = visual.TextStim(self.win, pos=[0,0], height=1, wrapWidth=40,      # pos=[.5, 0.1], height=0.03,
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
                    self.labels, self.diff_label, self.sim_label)
                self.win.flip()

                # Close the window if they hit 'escape'
                if event.getKeys(['escape']):
                    core.quit()
                    
            t1 = timer.getTime()
            if writeData:
                self.dataFile.write('%s,%s,%s,%s\n' %(A+'/'+choice1, B+'/'+choice2, self.rating.getRating(), t1-t0))

    def drawAll(self, *argv):
        for arg in argv:
            arg.draw()

if __name__ == '__main__': 
    b = BirdSimExp()
    b.instructions()
    b.trials(15, False)     # Run warmup trials (not recorded)
    b.trials(200, True)     # Run experiment trials (recorded)
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
    