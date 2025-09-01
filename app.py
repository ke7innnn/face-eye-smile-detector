import cv2

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("haarcascade_eye.xml")
smile = cv2.CascadeClassifier("haarcascade_smile.xml")
cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    face_detect = face.detectMultiScale(gray,1.1,5)
    
    for (x,y,w,h) in face_detect:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


        roi_gray = gray[y:y + h, x:x + w]
        roi = frame[y:y + h, x:x + w]

        eye_detect = eye.detectMultiScale(roi_gray,1.1,10)
        if len(eye_detect) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        smile_detect = smile.detectMultiScale(roi_gray,1.7,22)
        if len(smile_detect) > 0:
            cv2.putText(frame, "Smile Detected", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        

    cv2.imshow("Face detection",frame )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




