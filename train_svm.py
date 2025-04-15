from numpy import expand_dims, asarray
from sklearn.preprocessing import LabelEncoder, Normalizer
from deepface import DeepFace
from numpy import load
from joblib import dump
from sklearn.svm import SVC

data = load('dataset.npz')
testX_faces = data['arr_2']

def get_embeddings(face_pixels, model_name='Facenet'):
    embedding = DeepFace.represent(img_path=face_pixels, model_name=model_name, enforce_detection=False)
    return asarray(embedding[0]["embedding"])

trainX = [get_embeddings(face) for face in data['arr_0']]
testX = [get_embeddings(face) for face in testX_faces]

trainX, testX = asarray(trainX), asarray(testX)
trainy, testy = data['arr_1'], data['arr_3']

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

dump(model, 'svm_model.joblib')
print("Trained and saved model to disk.")
