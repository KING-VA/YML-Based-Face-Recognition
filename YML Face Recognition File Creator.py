import cv2
import os
import numpy as np
from PIL import Image
import shutil
from time import sleep

margin = 5

File_PATH = "C:\\Users\\user\\Documents\\Face Recognition\\Face Rec Training"
path = "C:\\Data\\User 1"

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("C:\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml");

def imagesgetter():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    
    count = 0
    
    face_id = input(' Enter user ID:\n  ')
    
    pic_PATH = "C:\\Users\\user\\Documents\\Face Recognition\\Face Rec Training\\Data\\User " + str(face_id)

    if not os.path.exists(pic_PATH):
        os.makedirs(pic_PATH)
    

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=15,
            )

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            cv2.imwrite(pic_PATH + "\\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y - margin:y+h+margin,x-margin:x+w+margin])
            cv2.imshow('image', img)
        sleep(1)    
        if count >= 30:
            print("30 Images Taken")
            break


    cam.release()
    cv2.destroyAllWindows()
    return face_id, pic_PATH
    
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for img in imagePaths:

        PIL_img = Image.open(img).convert('L')
        
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(img)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

def main():
    user, pic_PATH = imagesgetter()
    print("The Images have been saved at " + pic_PATH + "\n")
    faces,ids = getImagesAndLabels(pic_PATH)
    recognizer.train(faces, np.array(ids))
    YML_NAME = "User " + str(user)+ ' Trainer.yml'
    recognizer.write(YML_NAME)
    print("Created YML file with name " + YML_NAME + " at path: " + File_PATH + "\n")
    save_img = input("Enter 1 to delete the 30 .jpg images that were just created at " + pic_PATH + ".\n")
    if (str(save_img) == "1"):
        print("Removing files....")
        shutil.rmtree("C:\\Users\\user\\Documents\\Face Recognition\\Face Rec Training\\Data\\User " + str(user))
    print("YML file created and all operations completed. Have a nice Day.")

if __name__ == "__main__":
    main()
