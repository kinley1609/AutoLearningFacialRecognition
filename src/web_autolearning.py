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

# Supporting functions (unchanged)
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
    sorted_indices = np.argsort(similarities)[::-1]  # Sort similarities in descending order
    return sorted_indices, similarities
class VideoProcessor:
    def __init__(self, face_analyzer, model, raw_data_dir, model_path, register_name=None):
        self.face_analyzer = face_analyzer
        self.model = model
        self.raw_data_dir = raw_data_dir
        self.model_path = model_path
        self.high_confidence_images_captured = collections.defaultdict(int)
        self.RECOGNITION_THRESHOLD = 0.3
        self.user_input = None
        self.register_name = register_name
        self.captured_images = 0
        self.current_prediction = None  # Store current prediction for Streamlit
        self.current_prediction_index = 0  # Index to track next best prediction

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        faces = self.face_analyzer.get(img, max_num=1)
        self.current_prediction = None  # Reset for each frame

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            if self.register_name:
                # If registering, capture images of the person
                person_dir = os.path.join(self.raw_data_dir, self.register_name)
                create_directory(person_dir)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"{timestamp}_{self.captured_images}.jpg"
                img_path = os.path.join(person_dir, img_filename)
                cv2.imwrite(img_path, img)
                self.captured_images += 1
                st.write(f"Captured image {self.captured_images} for {self.register_name}")

                if self.captured_images >= 10:
                    st.success(f"Registration complete for {self.register_name}")
                    self.register_name = None
                    self.captured_images = 0

            else:
                sorted_indices, similarities = cosine_predict(embedding, self.model['embeddings'], self.model['labels'])
                best_match_index = sorted_indices[self.current_prediction_index]
                best_match_similarity = similarities[best_match_index]
                predicted_label = self.model['labels'][best_match_index]

                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{predicted_label}: {best_match_similarity:.2f}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if best_match_similarity > self.RECOGNITION_THRESHOLD:
                    self.current_prediction = (predicted_label, best_match_similarity, sorted_indices, similarities)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Face Recognition and Auto-Learning System")

    model_path = r'..\AutoLearningFacialRecognition\Models\new_facemodel.pkl'
    raw_data_dir = r'..\AutoLearningFacialRecognition\Dataset\FaceData\raw'

    face_analyzer = FaceAnalysis(name='buffalo_l',
                                 root=r'..\AutoLearningFacialRecognition\insightface_model')
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
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
        ["Attendance", "Register Face"]
    )

    if choice == "Attendance":
        if model is not None:
            st.subheader("Face Recognition")

            processor = VideoProcessor(face_analyzer, model, raw_data_dir, model_path)
            webrtc_ctx = webrtc_streamer(
                key="face-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=lambda: processor,
            )

            if webrtc_ctx.video_processor:
                prediction = webrtc_ctx.video_processor.current_prediction
                if prediction:
                    predicted_label, best_match_similarity, sorted_indices, similarities = prediction
                    st.write(f"Predicted name: {predicted_label} with similarity {best_match_similarity:.2f}")      

                    if st.button("This is me"):
                        st.success(f"Face confirmed as {predicted_label}")
                    
                    elif st.button("This is not me"):
                        st.error(f"Face not confirmed")
                        
                        # Move to the next prediction if available
                        if webrtc_ctx.video_processor.current_prediction_index + 1 < len(sorted_indices):
                            webrtc_ctx.video_processor.current_prediction_index += 1
                            next_best_match_index = sorted_indices[webrtc_ctx.video_processor.current_prediction_index]
                            next_best_match_similarity = similarities[next_best_match_index]
                            next_predicted_label = model['labels'][next_best_match_index]
                            st.write(f"Next predicted name: {next_predicted_label} with similarity {next_best_match_similarity:.2f}")
                        else:
                            st.write("No more predictions available.")
                    
                        if st.button("Confirm Correct Name"):
                            correct_name = st.text_input("Enter the correct name:")
                            if correct_name:
                                st.success(f"Face confirmed as {correct_name}")
                                model['labels'].append(correct_name)
                                model['embeddings'] = np.vstack((model['embeddings'], np.zeros((1, model['embeddings'].shape[1]))))
                                model['class_names'] = list(set(model['labels']))
                                with open(model_path, 'wb') as file:
                                    pickle.dump(model, file)
        else:
            st.error("Please train a model first.")

    elif choice == "Register Face":
        st.subheader("Register New Face")
        name = st.text_input("Enter name:")
        if st.button("Start Registration"):
            if name:
                st.write(f"Registering face for {name}. Please stay in front of the camera.")
                webrtc_streamer(
                    key="register-face",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                    video_processor_factory=lambda: VideoProcessor(face_analyzer, model, raw_data_dir, model_path, register_name=name),
                )
            else:
                st.error("Please enter a name to start registration.")

if __name__ == "__main__":
    main()


