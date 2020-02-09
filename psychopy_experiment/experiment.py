#!/usr/bin/env python
"""
Psychopy experiment for image selection
"""
from psychopy import core, visual, gui, data, event, monitors
from psychopy.tools.filetools import fromFile, toFile
import numpy, random, os
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
from psychopy.hardware import keyboard

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame
defaultKeyboard = keyboard.Keyboard()

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

# make a text file to save data
fileName = expInfo['ID'] + expInfo['dateStr']
path = os.getcwd()
dataFile = open(path+'/data/'+fileName+'.csv', 'w+')  # a simple text file with comma-separated-values
dataFile.write('leftIm,rightIm,userChoice\n')

# create window
win = visual.Window([1920, 1080],allowGUI=True, color=[0.2,0.2,0.2],
                    monitor='testMonitor', units='height')

# and some handy clocks to keep track of time
globalClock = core.Clock()
trialClock = core.Clock()

# display instructions and wait
msg1 = visual.TextStim(win, pos=[-.5,0.3], units='norm', height = 0.08, alignHoriz='left',
    text="You will be shown two images. \n\n\
Choose a value (1-8) on the scale to describe how similar they are.\n\n\
1 is least similar and 8 is most similar. \n\n\
Use the 1-8 keyboard keys to select the value. Then press Enter.")

msg2 = visual.TextStim(win, pos=[-.5, -0.3],units='norm', height=0.08, alignHoriz='left',
	text='Hit any key when you are ready to begin.')

msg1.draw()
msg2.draw()

win.flip()  #to show the messages
#pause until there's a keypress
event.waitKeys()

impath = path + '/images/'
all_birds = os.listdir(impath)

for i in range(50): # will continue the staircase until it terminates!
    # choose 3 random images of 3 types of birds
    [A, B] = random.sample(all_birds, 2)
    choice1 = random.choice(os.listdir(impath+A))
    choice2 = random.choice(os.listdir(impath+B))

    leftIm = visual.ImageStim(win, image=impath+A+'/'+choice1, 
        flipHoriz=True, pos=(-10,3), units='deg', size=(10,10))

    rightIm = visual.ImageStim(win, image=impath+B+'/'+choice2, 
        flipHoriz=True, pos=(10,3), units='deg', size=(10,10))

    msg = visual.TextStim(win, pos=[0, 0.6], units='norm', height=0.06, alignHoriz='center',
        text="Rate how similar the two images below are on a scale from 1-8. Then press Enter.")

    #rating = visual.RatingScale(win=win, name='rating', marker='triangle', 
     #   size=1.0, pos=[0.0, -0.4], low=1, high=8, labels=['similar', 'not similar'], scale='', markerStart='very similar')

    rating = visual.RatingScale(win=win,pos=[0, -.4], low=1, high=8, respKeys=['1','2','3','4','5','6','7','8'],
        acceptKeys=['return', 'space'], showAccept=False, textColor=[0.2,0.2,0.2] )

    labels = visual.TextStim(win, pos=[0,-.45], units='norm', height=0.06, alignHoriz='center',
        text="1        2       3        4       5       6       7       8")
    
    least_sim = visual.TextStim(win, pos=[-.5, -.45], units='norm',	height=.06, text='very different')
    most_sim = visual.TextStim(win, pos=[.5, -.45], units='norm',	height=.06, text='very similar')
    rating.reset()


    while rating.noResponse:
        leftIm.draw()
        rightIm.draw()
        msg.draw()
        rating.draw()
        labels.draw()
        least_sim.draw()
        most_sim.draw()
        win.flip()

        if event.getKeys(['escape']):
            core.quit()

    dataFile.write('%s,%s,%s\n' %(A+'/'+choice1, B+'/'+choice2, rating.getRating()))

dataFile.close()

win.flip()
event.waitKeys()  # wait for participant to respond

win.close()
core.quit()