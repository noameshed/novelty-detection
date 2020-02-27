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
            
        # Make a new csv and txt files to save data
        fileName = self.expInfo['Participant ID'] + '_PT=' + str(prompt_type) + '_' + self.expInfo['dateStr']
        self.dataFile = open(self.path+'/data/'+fileName+'.csv', 'w+')  # a simple text file with comma-separated-values
        self.dataFile.write('leftIm,rightIm,userChoice,responseTime\n')
        self.surveyFile = open(self.path+'/data/'+fileName+'.txt', 'w+')

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
        try:  # try to get a previous parameters file
            expInfo = fromFile(os.getcwd() + '/lastParams.pickle')
        except:  # if not there then use a default set
            expInfo = {'Participant ID':'', 'Experimenter ID':''}
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
        msg = visual.TextStim(self.win, pos=[0, 0], height=1, wrapWidth=40,                 # [0.5, 0.1]    0.03
            text=
            "In this experiment, you will be shown pairs of "+self.prompt+".\n\n\
Choose a value (1-7) on the scale to describe how similar they are.\n\n\
1 is least similar and 7 is most similar. Use the 1-7 keyboard keys to select the value.\n\n\
Then press Enter. Press any key when you are ready to begin.")
        msg.draw()

        self.win.flip()     # to show the messages
        event.waitKeys()    # pause until there's a keypress

    def getBirdPair(self):
        # Select a pair of birds to show the participant
        # Randomly select a difference bin to choose from

        d = numpy.random.choice(self.data)

        # Randomly select a pair of birds
        idx = numpy.random.choice(len(d))
        row = d[idx].split(',')
        A = row[0].split('_')[0]
        B = row[1].split('_')[0]

        # Get the paths to the bird images
        #[A, B] = random.sample(self.all_birds, 2)
        path1 = A + '.jpg'
        path2 = B + '.jpg'
            
        # Randomize the order in which the images are shown
        impaths = [path1, path2]
        random.shuffle(impaths)
        return impaths

    def trials(self, n, writeData):
        # Show message telling the participant they are starting a warmup trial
        msg = visual.TextStim(self.win, pos=[0,0], height=1, wrapWidth=40,
                text="You will now begin a practice round. You will see pairs of images and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")
        if writeData:   # Show messages telling the participant they are about to begin the real experiment
            msg = visual.TextStim(self.win, pos=[0,0], height=1, wrapWidth=40,    
                text="You will now begin the experiment. You will see pairs of images and choose a value (1-7) to decide how similar they are.\n\nPress any key when you are ready to begin.")
            
        msg.draw()
        self.win.flip()
        event.waitKeys()

        # Initialize the response timer 
        timer = core.Clock()

        for i in range(n): # 100 birds is about 10 minutes
            # choose an image pair from the precomputed list of image pairs
            [path1, path2] = self.getBirdPair()

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
                self.dataFile.write('%s,%s,%s,%s\n' %(path1, path2, self.rating.getRating(), t1-t0))

    def survey(self):
        # Ask the participant for some information
        title = visual.TextStim(self.win, pos=[0,12], height=1, wrapWidth=40,
            text='Survey (press Enter when done)')
        prompt = visual.TextStim(self.win, pos=[0,10], height=1, wrapWidth=40,
            text='What criteria, or rules, did you use to determine similarity and difference?')
        
        txt = ''
        msg = visual.TextStim(self.win, pos=[-19,0], height=1, wrapWidth=38, alignHoriz='left',
                text=txt)
        box = visual.Rect(self.win, pos=[0,0], width=40, height=10 )
        #msg = visual.TextBox(self.win, pos=[0,0], font_size=12, font_color=[-1,-1,-1],
        #    size=(10,6))
        self.drawAll(title, prompt, msg, box)
        self.win.flip()
        cap = False
        # Get user input and update textbox
        while True:
            keys = event.waitKeys()

            if 'return' in keys or 'escape' in keys:
                self.thankyou()

            # Fix keys printing character names
            if 'space' in keys:
                keys = ' '
            elif 'backspace' in keys:
                txt = txt[:-1]
                keys = ''
            elif 'apostrophe' in keys:
                keys = '\''
            elif 'comma' in keys:
                keys = ','
            elif 'period' in keys:
                keys = '.'
            elif 'backslash' in keys:
                keys = '\\'
            elif 'slash' in keys:
                keys = '/'
            elif 'escape' in keys:
                keys = ''

            #TODO: What other keys did I miss?

            if cap:
                keys[0] = keys[0].upper()
                cap = False
            if 'lshift' in keys or 'rshift' in keys:
                cap = True
                keys = ''

            txt += txt.join(keys)
            msg.setText(txt)
            self.drawAll(title, prompt, msg, box)
            self.win.flip()

        # Save response into text file:
        self.surveyFile.write(txt)

        #TODO: Why  doesn't this save the file properly?

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
    b.trials(15, True)     # Run warmup trials (recorded)
    b.trials(300, True)     # Run experiment trials (recorded)
    b.thankyou()