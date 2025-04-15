from numpy import load, asarray, savez_compressed
from deepface import DeepFace

# get the face embedding for one face
def get_embedding(face_pixels, model_name='Facenet'):
    # Dùng DeepFace để tạo embedding
    embedding = DeepFace.represent(img_path=face_pixels, model_name=model_name, enforce_detection=False)
    return asarray(embedding[0]["embedding"])

# load the face dataset
data = load('dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

# convert each face in the train set to an embedding
newTrainX = [get_embedding(face) for face in trainX]
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# convert each face in the test set to an embedding
newTestX = [get_embedding(face) for face in testX]
newTestX = asarray(newTestX)
print(newTestX.shape)

# save arrays to one file in compressed format
savez_compressed('embeddings.npz', newTrainX, trainy, newTestX, testy)
