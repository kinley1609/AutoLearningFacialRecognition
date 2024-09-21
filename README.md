# QAi Auto Learning System in Facial Recognition
This program is an product of the collaboration between QAi and the Team 6 of COS40005.

The following program is a project for the Auto Learning system in Facial Recognition.
The program is designed to recognize faces in real-time and learn new faces as it encounters them. The program uses a pre-trained model for face recognition and can be trained on new faces as needed. The program is designed to be user-friendly and easy to use. Each time the program is used, the model improves itself by re-training based on new and improved dataset.
# Installation
## Clone the Github repo:
```bash
git clone https://github.com/kinley1609/AutoLearningFacialRecognition
```
## Install the requirement:
```bash
pip install -r requirements.txt
```
# Usage
## Create 2 directories for Dataset and Model ( if not exist )
```bash
mkdir Dataset/FaceData/processed Dataset/FaceData/raw Models
```
## Pre-train Model for Face Recognition:
Download the pre-trained model from the following link and save it in the Models directory:
- https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
## Preparation:
- Prepare the Models directory with the pre-trained model before running the program.
```bash
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
```
# Run the program:
```bash
python ./src/autolearning.py

```
