import cv2, time
import numpy as np
from FullConvolutionModel import FCModel
from grabscreen import grab_screen

model = FCModel()
model.load_weights("TEST_ALL_CONV.model")

width = 801
height = 586

SCREEN_REGION = ((1920//2) - (1280//2), (1080//2) - (720//2) + 15, (1920//2) + (1280//2), (1080//2) + (720//2))
MODEL_SIZE = 224 # VGG: Image Size model requires
#MODEL_SIZE = 299 # Inceptionv3: Image Size model requires
NEW_SIZE = (320, 160)
previousTime = time.time()

def showFrameTime():
    global previousTime
    print(time.time() - previousTime)
    previousTime = time.time()
    
def preprocessInput(inputData):
    newData = inputData / 255
##    newData = newData - 0.5
##    newData = newData * 2

    return newData
    
while(True):
    screen = grab_screen(SCREEN_REGION)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    modelInput = cv2.resize(screen, NEW_SIZE)
    outputImage = np.array(modelInput)
    modelInput = np.expand_dims(modelInput, axis=0)
    modelInput = preprocessInput(modelInput)
    prediction = model.predict(modelInput)[0] * 255
    prediction = prediction.astype(np.uint8)

    #red = np.zeros_like(prediction).astype(np.int8)
    #blue = np.zeros_like(prediction).astype(np.int8)
    #overlay = np.concatenate((red, prediction, blue), axis=2)
    ret,prediction = cv2.threshold(prediction,1,255,cv2.THRESH_BINARY)
    overlay = np.zeros((160, 320, 3))
    overlay[:, :, 1] = prediction
    overlay = overlay.astype(np.uint8)
    overlay = cv2.resize(overlay, (screen.shape[1], screen.shape[0]))
    
    overlay = cv2.add(overlay*2, screen)
    cv2.imshow("LaneNN", overlay)
    cv2.waitKey(2)

    showFrameTime()
