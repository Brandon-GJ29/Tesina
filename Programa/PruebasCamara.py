import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configura la webcam
cap = cv2.VideoCapture(0)

# Inicia FaceMesh
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convierte a RGB
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Vuelve a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape

                # Landmarks del ojo izquierdo (desde el punto de vista de la persona)
                left_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7]
                # Landmarks del ojo derecho (desde el punto de vista de la persona)
                right_eye_indices = [263, 387, 385, 362, 380, 373, 390, 249]

                # Dibuja los puntos del ojo izquierdo (verde)
                for idx in left_eye_indices:
                    point = face_landmarks.landmark[idx]
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Dibuja los puntos del ojo derecho (azul)
                for idx in right_eye_indices:
                    point = face_landmarks.landmark[idx]
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

        # Mostrar imagen
        cv2.imshow('Segmentación de Ojos', image)
        if cv2.waitKey(5) & 0xFF == ord('e'):
            break

cap.release()
cv2.destroyAllWindows()
