import argparse
import cv2
import numpy as np
import os
import pickle
import collections
import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_images_in_raw(raw_data_dir):
    total_images = 0
    for root, dirs, files in os.walk(raw_data_dir):
        total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return total_images

def capture_images(name, raw_data_dir, num_images=200):
    face_analyzer = FaceAnalysis(name='buffalo_l', root='./insightface_model')
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    image_count = 0
    person_dir = os.path.join(raw_data_dir, name)
    
    create_directory(person_dir)
    print(f"Saving images to: {person_dir}")
    
    while image_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        
        faces = face_analyzer.get(frame, max_num=1)
        
        if len(faces) == 1:
            img_path = os.path.join(person_dir, f"{image_count}.jpg")
            cv2.imwrite(img_path, frame)
            image_count += 1
            print(f"Captured image {image_count}/{num_images}")
        else:
            print("Please ensure only one face is visible")
        
        cv2.imshow('Capture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {image_count} images for {name}")

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
                print(f"Skipping {image_path}: detected {len(faces)} faces")
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
    
    print(f"Model saved with {len(model['class_names'])} classes")
    print(f"Class names: {model['class_names']}")
    
    class_counts = collections.Counter(labels)
    print("Number of samples per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    
    return model

def cosine_predict(embedding, embeddings, labels):
    similarities = cosine_similarity([embedding], embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[best_match_index]
    predicted_label = labels[best_match_index]
    return predicted_label, best_match_similarity

def recognize_and_train(face_analyzer, model, raw_data_dir, model_path):
    if model is None or 'embeddings' not in model or 'labels' not in model:
        print("No valid model found. Please train a model first.")
        return None

    embeddings = model['embeddings']
    labels = model['labels']
    
    cap = cv2.VideoCapture(0)
    high_confidence_images_captured = collections.defaultdict(int)
    
    RECOGNITION_THRESHOLD = 0.3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        print(f"Frame shape: {frame.shape}")
        print(f"Frame min: {np.min(frame)}, max: {np.max(frame)}")

        faces = face_analyzer.get(frame, max_num=1)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(embedding)}")
            print(f"Embedding min: {np.min(embedding)}, max: {np.max(embedding)}")
            
            predicted_label, similarity = cosine_predict(embedding, embeddings, labels)
            print(f"Predicted label: {predicted_label}, Similarity: {similarity}")
            
            if similarity > RECOGNITION_THRESHOLD:
                name = predicted_label
            else:
                name = "Unknown"
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {similarity:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if similarity > 0.7 and name != "Unknown":
                person_dir = os.path.join(raw_data_dir, name)
                create_directory(person_dir)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"{timestamp}_{high_confidence_images_captured[name]}.jpg"
                img_path = os.path.join(person_dir, img_filename)
                cv2.imwrite(img_path, frame)
                high_confidence_images_captured[name] += 1
                print(f"Captured high confidence image for {name}")
                print(f"Image saved at: {img_path}")
                
                total_images = count_images_in_raw(raw_data_dir)
                print(f"Total images in raw directory: {total_images}")
                
                if total_images >= 300:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Total images reached 300. Training model...")
                    model = train_model(raw_data_dir, model_path)
                    print("Model updated. Clearing old images...")
                    
                    for person_folder in os.listdir(raw_data_dir):
                        person_path = os.path.join(raw_data_dir, person_folder)
                        if os.path.isdir(person_path):
                            for file in os.listdir(person_path):
                                os.remove(os.path.join(person_path, file))
                    
                    print("Old images cleared. Resuming recognition...")
                    cap = cv2.VideoCapture(0)
                    high_confidence_images_captured.clear()
        
        cv2.imshow('Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    MODEL_PATH = 'Models/facemodel.pkl'
    RAW_DATA_DIR = 'Dataset/FaceData/raw'

    face_analyzer = FaceAnalysis(name='buffalo_l', root='./insightface_model')
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"Face analyzer model: {face_analyzer.models}")

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model_data = pickle.load(file)
        print("Custom Classifier loaded successfully")
        print(f"Loaded model type: {type(model_data)}")
        if isinstance(model_data, tuple):
            model, class_names = model_data
            if isinstance(model, dict):
                model['class_names'] = class_names
            else:
                print("Unexpected model format")
                model = None
        elif isinstance(model_data, dict):
            model = model_data
        else:
            print("Unknown model format")
            model = None
        
        if model is not None:
            print(f"Model keys: {model.keys()}")
            print(f"Class names: {model.get('class_names', [])}")
    else:
        print("No existing model found. Please train a model first.")
        model = None

    while True:
        print("\nChoose an option:")
        print("1. Attendance")
        print("2. Register Face")
        print("3. Train Model")
        print("4. Attendance and Train")
        print("q. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            if model is not None:
                model = recognize_and_train(face_analyzer, model, RAW_DATA_DIR, MODEL_PATH)
            else:
                print("Please train a model first.")
        elif choice == '2':
            name = input("Enter name: ")
            capture_images(name, RAW_DATA_DIR)
        elif choice == '3':
            model = train_model(RAW_DATA_DIR, MODEL_PATH)
        elif choice == '4':
            if model is not None:
                model = recognize_and_train(face_analyzer, model, RAW_DATA_DIR, MODEL_PATH)
            else:
                print("Please train a model first.")
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()