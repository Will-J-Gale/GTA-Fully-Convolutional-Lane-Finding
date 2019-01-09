import numpy as np
import cv2, time
from grabscreen import grab_screen
from LaneNNModel import LaneNNModel

def showFrameTime():
    global previousTime
    print(time.time() - previousTime)
    previousTime = time.time()
    
def preprocessInput(image):
    newImage = cv2.resize(image, MODEL_INPUT_SIZE)
    newImage = np.expand_dims(newImage, axis=0)
    newImage = newImage / 255
    
    return newImage

def postProcessOutput(image):
    #Bring image from 0-1 to 0-255 and remove last axis
    newImage = image * 255
    newImage = newImage.astype(np.uint8)
    newImage = np.squeeze(newImage, axis=2)

    #Create blank image and add image to the GREEN channel
    overlay = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3))
    overlay[:, :, 1] = newImage
    overlay = overlay.astype(np.uint8)
    overlay = cv2.resize(overlay, NEW_SIZE, interpolation=cv2.INTER_NEAREST)

    return overlay

#Screen co-ordinate variables
#------------------------------------------------------------------------

width = 1920 - 1
height = 1080

gameWidth = 1274 
gameHeight = 714

xOffset = 2
yOffset = 17

gameStartX = ((width//2) - (gameWidth//2)) + xOffset
gameStartY = ((height//2) - (gameHeight//2)) + yOffset
gameEndX = ((width//2) + (gameWidth//2)) + xOffset
gameEndY = ((height//2) + (gameHeight//2)) + yOffset
SCREEN_REGION = (gameStartX, gameStartY, gameEndX, gameEndY)

#Size of image going into Neural Network
NEW_SIZE = (800, 450)
MODEL_INPUT_SIZE = (320, 160)
#Lane Finding Model
#------------------------------------------------------------------------
MODEL_WEIGHTS = "LaneNN.model"
model = LaneNNModel()
model.load_weights(MODEL_WEIGHTS)
AI_ENABLED = True

#Start Frame time clock
previousTime = time.time()

if __name__ == "__main__":

    while(True):

        #Check for keyboard interrupt
        try:
            start = time.time()
            screen = grab_screen(SCREEN_REGION)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            if(AI_ENABLED):
                modelInput = preprocessInput(screen)
                prediction = model.predict(modelInput)[0]
                overlay = postProcessOutput(prediction)
                
                screen = cv2.resize(screen, NEW_SIZE)
                overlay = cv2.add(overlay, screen)
             
                cv2.imshow("Lane", overlay)
                key = cv2.waitKey(2)
                #showFrameTime()
        except KeyboardInterrupt:
            print("Closing")
            break
        
    cv2.destroyAllWindows()
