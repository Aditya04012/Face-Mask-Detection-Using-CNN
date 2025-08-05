import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model('Face_mask_98.90%.h5')


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

IMG_SIZE = 128
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x = max(0, x)
            y = max(0, y)
            w = min(iw - x, w)
            h = min(ih - y, h)

            
            face = frame[y:y + h, x:x + w]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue  

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_rgb = face_rgb / 255.0
            input_img = np.expand_dims(face_rgb, axis=0)

            
            prediction = model.predict(input_img)[0][0]

            if prediction < 0.5:
                label = "With Mask"
                color = (0, 255, 0)
            else:
                label = "Without Mask"
                color = (0, 0, 255)

           
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

   
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
