import cv2
from random import choice
from numpy import expand_dims, asarray
from sklearn.preprocessing import LabelEncoder, Normalizer
from numpy import load
from joblib import load as load_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace


def extract_face(pixels, required_size=(160, 160)):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        print("No face detected in the frame")
        return None
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    face = Image.fromarray(face)
    face = face.resize(required_size)
    face_array = asarray(face)
    return face_array


def get_embeddings(face_pixels, model_name='Facenet'):
    embedding = DeepFace.represent(img_path=face_pixels, model_name=model_name, enforce_detection=False)
    return asarray(embedding[0]["embedding"])


trainy = load('dataset.npz')['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

model = load_model('svm_model.joblib')
print("Loaded model from disk.")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_bounding = extract_face(rgb_frame)

    if face_bounding is not None:
        face_emb = asarray(get_embeddings(face_bounding))
        samples = expand_dims(face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        cv2.putText(frame, f'Predicted: {predict_names[0]} ({class_probability:.2f}%)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()
