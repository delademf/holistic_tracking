#Import dependencies
import cv2 as cv
import mediapipe as mp
import numpy as np

#setting up media pipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#Rescale function
def rescaleframe(frame,scale=0.5):
    width = int(frame.shape[1]*0.5)
    height = int(frame.shape[0]*0.5)
    dimensions =(width,height)

    cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

#Adding Styling
mp_drawing.DrawingSpec(color=(0,255,0),thickness= 2,circle_radius=2)


# Get webcam in realtime
capture =cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence = 0.5) as holistic:
#the last holistic is a holistic namimng convention = term
    while True:
        isTrue,frame= capture.read()
        
        # Recolor feed
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        #print the position of the holistic feed
        result = holistic.process(imgRGB)
        # print(result.pose_landmarks)

        #recolor RGB to BGR
        imgBGR = cv.cvtColor(imgRGB,cv.COLOR_RGB2BGR)

        #drawing facial landmarks
        mp_drawing.draw_landmarks(imgBGR,result.face_landmarks,mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(200,167,123),thickness= 2,circle_radius=1),
                                   mp_drawing.DrawingSpec(color=(211,0,120),thickness= 2,circle_radius=1) )#its suppose to be holistic.FACE_CONNECTIONS so check it

        #drawing pose landmarks
        mp_drawing.draw_landmarks(imgBGR,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                  #the first line colors the landmarks and the second the connections
                                   mp_drawing.DrawingSpec(color=(0,255,0),thickness= 2,circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(255,0,255),thickness= 2,circle_radius=2))

        #drawing right hand landmarks
        mp_drawing.draw_landmarks(imgBGR,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0),thickness= 2,circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness= 2,circle_radius=2)
                                  )
        
        #drawing left hand landmarks
        mp_drawing.draw_landmarks(imgBGR,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0),thickness= 2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,0,255),thickness= 2,circle_radius=2)
                                )

        cv.imshow('raw webcam feed',imgBGR)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()


