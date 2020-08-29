import cv2

# For no Face Detected
noFaceFound_Text = 'No Faces Found'
noFaceFound_TextColor = (0, 0, 255)

# For Mask Detected
maskFound_GreenColor = (0, 255, 0)
maskFound_Text = 'Mask Found, You\'re safe'
maskFound_TextColor = (255, 255, 255)

# For No Mask Detected
noMaskFound_RedColor = (0, 0, 255)
noMaskFound_Text = 'No Mask Found, ALERT!!!'
noMaskFound_TextColor = (255, 255, 255)

def readFiles():
    # Frontal face Cascade file provided by OpenCV
    faceCascade = cv2.CascadeClassifier('XML files\datasets_13405_18147_haarcascade_frontalface_default.xml')

    # My custom trained Cascade file
    maskCascade = cv2.CascadeClassifier('XML files\haarcascade_mask.xml')

    return faceCascade, maskCascade

def captureVideo():
    # Using my Default WebCam in my Computer
    capture = cv2.VideoCapture(0)

    # Width of the window
    capture.set(3, 640)

    # Height of the window
    capture.set(4, 480)

    # Brightness of the the window
    capture.set(10, 150)

    return capture

# Function Call for readFiles
faceCascade, maskCascade = readFiles()

# Fucntion Call for captureVideo
capture = captureVideo()

while True:
    # Reading  the images Frame by Frame
    returnValue, image = capture.read()

    # Converting into BGR image to GrayScale image
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecting Face in the frames
    face = faceCascade.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=4)

    # Detecting Mask in the frames
    mask = maskCascade.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=2)

    # Checking if faces and masks are found
    #print(face, mask)

    # If no Face is Found
    if face==():
        cv2.putText(image, noFaceFound_Text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=noFaceFound_TextColor, thickness=2)

    # If Face is Found
    else:
        for rect in face:

            # Getting the X coordinate, Y coordinate, Width and Height from the Rectangle
            x, y, w, h = rect

            # If Mask is Found
            if mask!=():
                # Rectangle Bounding the Face
                cv2.rectangle(image, pt1=(x-20, y-20), pt2=(x+w+20, y+h+20), color=maskFound_GreenColor, thickness=3)

                # Rectangle Bounding the Text
                cv2.rectangle(image, (x-20, y-50), (x+w+20, y-20), maskFound_GreenColor, cv2.FILLED)

                # The Text at the top of the Face
                cv2.putText(image, maskFound_Text, org=(x-20, y-25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=maskFound_TextColor, thickness=2)

            # If Mask is Not Found
            else:
                cv2.rectangle(image, pt1=(x - 20, y - 20), pt2=(x + w + 20, y + h + 20), color=noMaskFound_RedColor, thickness=3)
                cv2.rectangle(image, (x - 20, y - 50), (x + w + 20, y - 20), noMaskFound_RedColor, cv2.FILLED)
                cv2.putText(image, noMaskFound_Text, org=(x - 20, y - 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=noMaskFound_TextColor, thickness=2)

    # Displaying the images Frame by Frame
    cv2.imshow('Output', image)

    # The window waits for the 'Esc' Key to break the loop (ASCII value of Esc = 27)
    if cv2.waitKey(1) & 0xff == 27:
        break

# Closing all the Windows
cv2.destroyAllWindows()
capture.release()
