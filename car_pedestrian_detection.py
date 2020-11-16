import cv2

body_classifier=cv2.CascadeClassifier("./Haarcascades/haarcascade_fullbody.xml")
car_classifier = cv2.CascadeClassifier("./Haarcascades/haarcascade_car.xml")

video=cv2.VideoCapture('./images/walking.avi')
while video.isOpened():
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pedestrains = body_classifier.detectMultiScale(gray_frame, 1.2, 3)
    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("video playing", frame)
    if cv2.waitKey(1) == 13:
        break

video=cv2.VideoCapture('./images/cars.avi')


while video.isOpened():
    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cars = car_classifier.detectMultiScale(gray_frame, 1.2, 3)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("video playing", frame)
    if cv2.waitKey(1) == 13:
        break
video.release()
cv2.destroyAllWindows()