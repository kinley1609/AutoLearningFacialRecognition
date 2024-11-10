from __future__ import absolute_import, division, print_function

import tensorflow as tf
import argparse
import facenet
import imutils
import os
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import subprocess
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images(name, num_images=100):
    cap = cv2.VideoCapture(0)
    image_count = 0
    create_directory(f"./Dataset/FaceData/raw/{name}")
    
    while image_count < num_images:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        cv2.imshow('Register Face', frame)
        img_path = f"./Dataset/FaceData/raw/{name}/{image_count}.jpg"
        cv2.imwrite(img_path, frame)
        image_count += 1
        print(f"Captured image {image_count}/{num_images}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def recognize_faces(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names):
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

    cap = cv2.VideoCapture(0)
    high_confidence_images_captured = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]
        try:
            if faces_found > 1:
                cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
            elif faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        if best_class_probabilities > 0.7:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            name = class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)

                            if best_class_probabilities > 0.9:
                                if high_confidence_images_captured < 10:
                                    create_directory(f"./Dataset/FaceData/raw/{best_name}")
                                    img_path = f"./Dataset/FaceData/raw/{best_name}/{len(os.listdir(f'./Dataset/FaceData/raw/{best_name}'))}.jpg"
                                    cv2.imwrite(img_path, frame)
                                    high_confidence_images_captured += 1
                                    print(f"Captured high confidence image {high_confidence_images_captured} for {best_name}")

                                if high_confidence_images_captured >= 10:
                                    print(f"Captured 10 high confidence images for {best_name}. Exiting...")
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return

                        else:
                            name = "Unknown"

        except Exception as e:
            print("Error:", e)
            pass

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def align_faces(raw_dir, processed_dir):
    subprocess.run([
        'python', 'src/align_dataset_mtcnn.py',
        raw_dir,
        processed_dir,
        '--image_size', '160',
        '--margin', '32',
        '--random_order',
        '--gpu_memory_fraction', '0.25'
    ])

def train_model(processed_dir, model_path, classifier_path):
    subprocess.run([
        'python', 'src/classifier.py',
        'TRAIN',
        processed_dir,
        model_path,
        classifier_path,
        '--batch_size', '1000'
    ])

def recognize_and_train(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names, raw_data_dir, processed_data_dir, model_path, classifier_path):
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

    cap = cv2.VideoCapture(0)
    high_confidence_images_captured = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]
        try:
            if faces_found > 1:
                cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
            elif faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        if best_class_probabilities > 0.7:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            name = class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)

                            if best_class_probabilities > 0.9:
                                if high_confidence_images_captured < 10:
                                    create_directory(f"./Dataset/FaceData/raw/{best_name}")
                                    img_path = f"./Dataset/FaceData/raw/{best_name}/{len(os.listdir(f'./Dataset/FaceData/raw/{best_name}'))}.jpg"
                                    cv2.imwrite(img_path, frame)
                                    high_confidence_images_captured += 1
                                    print(f"Captured high confidence image {high_confidence_images_captured} for {best_name}")

                                if high_confidence_images_captured >= 10:
                                    print(f"Captured 10 high confidence images for {best_name}.")
                                    high_confidence_images_captured = 0  # Reset counter for next round

                                    # Check if raw image count is >= 100
                                    total_images = sum([len(files) for r, d, files in os.walk(raw_data_dir)])
                                    if total_images >= 100:
                                        print(f"Total images in raw data: {total_images}. Training model...")
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        align_faces(raw_data_dir, processed_data_dir)
                                        train_model(processed_data_dir, model_path, classifier_path)
                                        for root, dirs, files in os.walk(raw_data_dir):
                                            for file in files:
                                                os.remove(os.path.join(root, file))
                                        print("Training completed and raw data cleared. Resuming attendance...")
                                        cap = cv2.VideoCapture(0)

                        else:
                            name = "Unknown"

        except Exception as e:
            print("Error:", e)
            pass

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def evaluate_model(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names, test_data_dir):
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160

    # Create MTCNN
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

    total_images = sum([len(files) for r, d, files in os.walk(test_data_dir)])
    print(f"Total images: {total_images}")

    y_true = []
    y_pred = []
    confidences = []

    # Process each class
    for class_name in class_names:
        test_images_path = os.path.join(test_data_dir, class_name)
        if not os.path.exists(test_images_path):
            print(f"Warning: Directory not found for class {class_name}")
            continue

        print(f"\nProcessing class: {class_name}")
        
        # Process each image in the class directory
        for image_file in os.listdir(test_images_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(test_images_path, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            try:
                # Detect face
                bounding_boxes, _ = align.detect_face.detect_face(
                    frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                if len(bounding_boxes) < 1:
                    print(f"No face detected in {image_file}")
                    continue

                # Use the first detected face
                det = bounding_boxes[0, 0:4]
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = max(det[0], 0)
                bb[1] = max(det[1], 0)
                bb[2] = min(det[2], frame.shape[1])
                bb[3] = min(det[3], frame.shape[0])

                # Crop and process face
                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                  interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                
                # Get embedding
                feed_dict = {
                    images_placeholder: scaled_reshape,
                    phase_train_placeholder: False
                }
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                # Make prediction
                predictions = model.predict_proba(emb_array)
                best_class_index = np.argmax(predictions[0])
                confidence = predictions[0][best_class_index]
                predicted_name = class_names[best_class_index]

                y_true.append(class_name)
                y_pred.append(predicted_name)
                confidences.append(confidence)

                print(f"Image: {image_file}, Predicted: {predicted_name}, "
                      f"Confidence: {confidence:.3f}")

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue

    # Calculate metrics
    if len(y_true) > 0:
        print("\nEvaluation Results:")
        print("------------------")
        
        # Calculate and print accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate and print precision, recall, and F1 score
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        print("\nConfusion Matrix:")
        print("----------------")
        print("True labels (rows) vs Predicted labels (columns)")
        for i, row in enumerate(cm):
            print(f"{class_names[i]}: {row}")

        # Print detailed classification report
        print("\nClassification Report:")
        print("---------------------")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # Print average confidence
        avg_confidence = np.mean(confidences)
        print(f"\nAverage Confidence: {avg_confidence:.4f}")

    else:
        print("No valid predictions were made. Check your test data and face detection parameters.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()

    CLASSIFIER_PATH = './Models/facemodel.pkl'
    FACENET_MODEL_PATH = './Models/20180402-114759.pb'
    RAW_DATA_DIR = './Dataset/FaceData/raw'
    PROCESSED_DATA_DIR = './Dataset/FaceData/processed'
    TEST_DATA_DIR = './Dataset/FaceData/test'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            while True:
                print("Choose an option:")
                print("1. Attendance")
                print("2. Register Face")
                print("3. Train Model")
                print("4. Attendance and Train")
                print("5. Evaluate Model")
                print("q. Quit")
                choice = input("Enter your choice: ")

                if choice == '1':
                    recognize_faces(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names)
                elif choice == '2':
                    name = input("Enter name: ")
                    create_directory(f"./Dataset/FaceData/raw/{name}")
                    capture_images(name)
                elif choice == '3':
                    align_faces(RAW_DATA_DIR, PROCESSED_DATA_DIR)
                    train_model(PROCESSED_DATA_DIR, FACENET_MODEL_PATH, CLASSIFIER_PATH)
                elif choice == '4':
                    recognize_and_train(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names, RAW_DATA_DIR, PROCESSED_DATA_DIR, FACENET_MODEL_PATH, CLASSIFIER_PATH)
                elif choice == '5':
                    evaluate_model(sess, images_placeholder, embeddings, phase_train_placeholder, model, class_names, TEST_DATA_DIR)
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice. Please try again.")

main()
