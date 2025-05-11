import cv2

cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read() #Abre  la camara y empieza el webcam 
    if not ret:
        break

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('e'): # Cuando  se preciona la telca e finaliza
        break

cap.release()
cv2.destroyAllWindows()
