import streamlit as st
import cv2
import numpy as np
import os
import pickle
import collections
import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Các hàm hỗ trợ (giữ nguyên)
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_images_in_raw(raw_data_dir):
    total_images = 0
    for root, dirs, files in os.walk(raw_data_dir):
        total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return total_images

def train_model(data_dir, model_path):
    face_analyzer = FaceAnalysis(name='buffalo_l')
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    embeddings = []
    labels = []
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            
            faces = face_analyzer.get(image, max_num=1)
            if len(faces) != 1:
                st.warning(f"Skipping {image_path}: detected {len(faces)} faces")
                continue
            
            face = faces[0]
            embedding = face.embedding
            
            embeddings.append(embedding)
            labels.append(person_name)
    
    model = {
        'embeddings': np.array(embeddings),
        'labels': labels,
        'class_names': list(set(labels))
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    st.success(f"Model saved with {len(model['class_names'])} classes")
    st.write(f"Class names: {model['class_names']}")
    
    class_counts = collections.Counter(labels)
    st.write("Number of samples per class:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")
    
    return model

def cosine_predict(embedding, embeddings, labels):
    similarities = cosine_similarity([embedding], embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[best_match_index]
    predicted_label = labels[best_match_index]
    return predicted_label, best_match_similarity

class VideoProcessor:
    def __init__(self, face_analyzer, model, raw_data_dir, model_path):
        self.face_analyzer = face_analyzer
        self.model = model
        self.raw_data_dir = raw_data_dir
        self.model_path = model_path
        self.high_confidence_images_captured = collections.defaultdict(int)
        self.RECOGNITION_THRESHOLD = 0.3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        faces = self.face_analyzer.get(img, max_num=1)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            predicted_label, similarity = cosine_predict(embedding, self.model['embeddings'], self.model['labels'])
            
            if similarity > self.RECOGNITION_THRESHOLD:
                name = predicted_label
            else:
                name = "Unknown"
            
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{name}: {similarity:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if similarity > 0.7 and name != "Unknown":
                person_dir = os.path.join(self.raw_data_dir, name)
                create_directory(person_dir)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"{timestamp}_{self.high_confidence_images_captured[name]}.jpg"
                img_path = os.path.join(person_dir, img_filename)
                cv2.imwrite(img_path, img)
                self.high_confidence_images_captured[name] += 1
                st.write(f"Captured high confidence image for {name}")
                
                total_images = count_images_in_raw(self.raw_data_dir)
                st.write(f"Total images in raw directory: {total_images}")
                
                if total_images >= 100:
                    st.write("Total images reached 100. Training model...")
                    self.model = train_model(self.raw_data_dir, self.model_path)
                    st.success("Model updated. Clearing old images...")
                    
                    for person_folder in os.listdir(self.raw_data_dir):
                        person_path = os.path.join(self.raw_data_dir, person_folder)
                        if os.path.isdir(person_path):
                            for file in os.listdir(person_path):
                                os.remove(os.path.join(person_path, file))
                    
                    st.write("Old images cleared. Resuming recognition...")
                    self.high_confidence_images_captured.clear()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Face Recognition and Auto-Learning System")

    MODEL_PATH = 'Models/facemodel.pkl'
    RAW_DATA_DIR = 'Dataset/FaceData/raw'

    face_analyzer = FaceAnalysis(name='buffalo_l', root='./insightface_model')
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model_data = pickle.load(file)
        st.success("Custom Classifier loaded successfully")
        if isinstance(model_data, dict):
            model = model_data
        else:
            st.warning("Unknown model format")
            model = None
        
        if model is not None:
            st.write(f"Model keys: {model.keys()}")
            st.write(f"Class names: {model.get('class_names', [])}")
    else:
        st.warning("No existing model found. Please train a model first.")
        model = None

    choice = st.sidebar.selectbox(
        "Choose an option",
        ["Attendance", "Register Face", "Train Model", "Attendance and Train"]
    )

    if choice == "Attendance":
        if model is not None:
            st.subheader("Face Recognition")
            webrtc_streamer(
                key="face-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=lambda: VideoProcessor(face_analyzer, model, RAW_DATA_DIR, MODEL_PATH),
                async_processing=True,
            )
        else:
            st.error("Please train a model first.")

    elif choice == "Register Face":
        st.subheader("Register New Face")
        name = st.text_input("Enter name:")
        if st.button("Capture Images"):
            # Implement Streamlit version of capture_images
            st.warning("This feature is not implemented in the Streamlit version yet.")

    elif choice == "Train Model":
        st.subheader("Train Model")
        if st.button("Start Training"):
            model = train_model(RAW_DATA_DIR, MODEL_PATH)
            st.success("Model trained successfully!")

    elif choice == "Attendance and Train":
        if model is not None:
            st.subheader("Face Recognition and Auto-Learning")
            webrtc_streamer(
                key="face-recognition-and-train",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=lambda: VideoProcessor(face_analyzer, model, RAW_DATA_DIR, MODEL_PATH),
                async_processing=True,
            )
        else:
            st.error("Please train a model first.")

if __name__ == "__main__":
    main()