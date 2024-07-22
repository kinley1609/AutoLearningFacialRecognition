# train_new_arcface_model.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
from sklearn.svm import SVC

def load_face_analyzer():
    face_analyzer = FaceAnalysis(name='buffalo_l', root='./insightface_model')
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer

def get_face_embedding(face_analyzer, img):
    faces = face_analyzer.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

def train_new_model(data_dir, model_path):
    face_analyzer = load_face_analyzer()
    
    embeddings = []
    labels = []
    class_names = []
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        if person_name not in class_names:
            class_names.append(person_name)
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Could not read image: {image_path}")
                continue
            
            embedding = get_face_embedding(face_analyzer, img)
            
            if embedding is None:
                print(f"No face detected in {image_path}")
                continue
            
            embeddings.append(embedding)
            labels.append(person_name)
    
    # Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump((model, class_names), f)
    
    print(f"New model trained and saved successfully to {model_path}")
    print(f"Class names: {class_names}")

if __name__ == "__main__":
    DATA_DIR = 'Dataset/FaceData/raw'  # Đường dẫn đến thư mục chứa dữ liệu mới của bạn
    NEW_MODEL_PATH = 'Models/new_facemodel.pkl'  # Đường dẫn để lưu tệp pkl mới
    
    train_new_model(DATA_DIR, NEW_MODEL_PATH)