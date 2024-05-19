import cv2

car_classifier = cv2.CascadeClassifier("cars.xml")
video = cv2.VideoCapture("video.mp4")
# if we want to capture video from any recording device
# video = cv2.VideoCapture(1)
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car = car_classifier.detectMultiScale(gray, 1.1, 9)
    # gray scale variable - gray
    # scale factor - parameter to specifies how much the image size is reduced at each image scale
    # 6 - minNeighours
    for (x, y, w, h) in car:
        plate = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (51, 51, 255), -2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Car', plate)
    frames = cv2.resize(frame, (600, 400))
    cv2.imshow('Car Detection System', frames)
    k = cv2.waitKey(30) & 0xff
    # if we press esc key, it will close all the window
    if k == 27:
        break
video.release()
cv2.destroyAllWindows()
