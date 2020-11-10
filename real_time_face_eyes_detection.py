import cv2

classifier = cv2.CascadeClassifier('../Haarcascades/haarcascade_frontalface_default.xml')
classifier_eyes= cv2.CascadeClassifier('../Haarcascades/haarcascade_eye.xml')


def facedetected(input ):
    gray_image = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    images = classifier.detectMultiScale(gray_image, 1.3, 5)
    if images is ():
        print(" no face detected")
    else:
        print(" the image contains a face or more ")
        for (x, y, w, h) in images:
            cv2.rectangle(input, (x, y), (x + w, y + h), (0, 0, 255), 3)
            gray_face = gray_image[y:y+h,x:x+w]
            colored_image = input[y:y+h,x:x+w]
            eyes = classifier_eyes.detectMultiScale(gray_face)
            for (ex,ey,ew,eh) in eyes:
                print("eyes detected")
                cv2.rectangle(input,(x+ex,y+ey),(x+ex+ew-10,y+ey+eh-10),(255,0,0),2)
    return input
pass

capture = cv2.VideoCapture(0)

while True:
    ret,frame=capture.read()
    cv2.imshow(" real time face detection ",facedetected(frame))
    if cv2.waitKey(1) == 13 :
        break


capture.release()
cv2.destroyAllWindows()
