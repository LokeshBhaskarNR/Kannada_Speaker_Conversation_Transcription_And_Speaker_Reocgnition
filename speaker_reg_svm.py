import os
import librosa
import numpy as np
import shutil
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import soundfile as sf
import joblib
import time

model_dir = "Speaker_Models"
os.makedirs(model_dir, exist_ok=True)

data_dir = "Segment_Audio"
segment_length = 1  # seconds
sampling_rate = 16000
n_mfcc = 13
max_frames = 100

label_encoder = LabelEncoder()

def segment_audio(audio_path, speaker_name, save_path):

    os.makedirs(save_path, exist_ok=True)
    
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(total_duration // segment_length)
    
    for i in range(num_segments):
        start_sample = i * sampling_rate * segment_length
        end_sample = start_sample + sampling_rate * segment_length
        segment = audio[start_sample:end_sample]
        segment_filename = os.path.join(save_path, f"{speaker_name}_{i}.wav")
        sf.write(segment_filename, segment, sampling_rate)
    
    print(f"Audio split into {num_segments} segments and saved in {save_path}")

def preprocess_audio(file_path):

    audio, sr = librosa.load(file_path, sr=sampling_rate, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    scaler = StandardScaler()
    mfccs = scaler.fit_transform(mfccs)
    
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_frames]
    
    return mfccs.T

def load_dataset(data_dir):

    features, labels = [], []
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    
    label_encoder.fit(class_names)
    
    for label in class_names:
        speaker_path = os.path.join(data_dir, label)
        for file in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, file)
            mfccs = preprocess_audio(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)
    
    return np.array(features), label_encoder.transform(labels), label_encoder

def retrain_model(audio_folder, status):

    X, y, label_encoder = load_dataset(data_dir)
    X = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("[INFO] --> Dataset loaded successfully!")
    print("length of X_train: ", len(X_train))
    print("length of y_train: ", len(y_train))
    print("length of X_test: ", len(X_test))
    print("length of y_test: ", len(y_test))
    
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_classifier.fit(X_train, y_train)
    
    model_path = os.path.join(model_dir, "svm_model.pkl")
    joblib.dump(svm_classifier, model_path)
    print("[INFO] --> Model saved successfully!") 
    
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_path)
    print("[INFO] --> Label Encoder saved successfully!")

    print("Model re-trained successfully!")
    
    if status == True:
        predictions_file = ("predicted_speakers.txt")
        with open(predictions_file, "w") as f:

            for file in os.listdir(audio_folder):
                if file.endswith(".wav"):  
                    file_path = os.path.join(audio_folder, file)
                    
                    features = preprocess_audio(file_path)
                    features = features.reshape(1, -1)
                    
                    predicted_label = svm_classifier.predict(features)[0]
                    predicted_class = label_encoder.inverse_transform([predicted_label])[0]

                    f.write(f"{predicted_class}\n")
                    print(f"[INFO] File: {file} → Predicted: {predicted_class}")

        print(f"[INFO] --> Predictions saved in: {predictions_file}")

def add_new_speaker(audio_path, speaker_name):

    speaker_folder = os.path.join(data_dir, speaker_name)
    if os.path.exists(speaker_folder):
        print("Speaker already exists! Try a different name.")
        return
    
    segment_audio(audio_path, speaker_name, speaker_folder)
    print(f"New speaker '{speaker_name}' added.")

