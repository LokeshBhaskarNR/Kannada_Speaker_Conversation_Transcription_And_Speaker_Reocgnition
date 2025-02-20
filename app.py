import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
import soundfile as sf
import joblib
from speaker_reg_svm import preprocess_audio, label_encoder, load_dataset
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import io
import base64
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder

from audio_to_text import transcribe_audio_folder
from speaker_diarize import (
    system_check, process_audio
)
from speaker_reg_svm import retrain_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATASET_FOLDER'] = 'Segment_Audio/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        num_speakers = request.form.get('num_speakers', 2)  # Default to 2
        try:
            num_speakers = int(num_speakers)
        except ValueError:
            return "Invalid number of speakers", 400

        if 'audio' in request.files and request.files['audio'].filename != '':
            file = request.files['audio']
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('process_audio_ajax', filename=filename, num_speakers=num_speakers))

        elif 'recordedAudio' in request.form and request.form['recordedAudio']:
            recorded_data = request.form['recordedAudio']
            audio_data = base64.b64decode(recorded_data.split(',')[1])  # Decode Base64 audio data
            filename = str(uuid.uuid4()) + ".wav"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            with open(filepath, "wb") as f:
                f.write(audio_data)

            return redirect(url_for('process_audio_ajax', filename=filename, num_speakers=num_speakers))

    return render_template('upload.html')

@app.route('/process_audio/<filename>/<int:num_speakers>')
def process_audio_ajax(filename, num_speakers):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': 'Audio file not found'}), 404

        # Process the audio
        system_check()
        process_audio(filepath, num_speakers)
        retrain_model("Speaker_Split_Audios", status=True)
        transcription_result = transcribe_audio_folder("Speaker_Split_Audios")
        
        return redirect(url_for('results', filename=filename, num_speakers=num_speakers))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

        

@app.route('/results/<filename>/<int:num_speakers>')
def results(filename, num_speakers):
    try:
        # Check if all required files exist
        required_files = {
            "speaker_timeline.txt": "Timeline data",
            "predicted_speakers.txt": "Speaker data",
            "audio_transcriptions.txt": "Transcription data"
        }
        
        for file_path, data_type in required_files.items():
            if not os.path.exists(file_path):
                return render_template('error.html', 
                                     message=f"Processing incomplete: Missing {data_type}"), 404

        # Read all files with proper error handling
        speaker_data = []
        
        with open("speaker_timeline.txt", "r", encoding="utf-8") as f:
            timelines = [line.strip() for line in f.readlines()]
            
        with open("predicted_speakers.txt", "r", encoding="utf-8") as f:
            speakers = [line.strip() for line in f.readlines()]
            
        with open("audio_transcriptions.txt", "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f.readlines()]

        # Combine the data
        for i in range(len(speakers)):
            speaker_data.append({
                "name": speakers[i],
                "timeline": timelines[i] if i < len(timelines) else "No timeline available",
                "transcription": transcriptions[i] if i < len(transcriptions) else "No transcription available"
            })

        if os.path.exists("speaker_timeline.txt"):
            os.remove("speaker_timeline.txt")
        if os.path.exists("predicted_speakers.txt"):
            os.remove("predicted_speakers.txt")
        if os.path.exists("audio_transcriptions.txt"):
            os.remove("audio_transcriptions.txt")


        for file in os.listdir("Speaker_Split_Audios"):
            file_path = os.path.join("Speaker_Split_Audios", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        print("[INFO] --> All files removed Successfully !")
        return render_template('results.html', 
                             filename=filename, 
                             num_speakers=num_speakers, 
                             speaker_data=speaker_data)
                             
    except Exception as e:
        return render_template('error.html', 
                             message=f"Error processing results: {str(e)}"), 500



def generate_confusion_matrix():
    svm_model = joblib.load("Speaker_Models/svm_model.pkl")
    label_encoder = joblib.load("Speaker_Models/label_encoder.pkl")
    try:
        
        X, y, label_encoder = load_dataset(data_dir)
        X = X.reshape(X.shape[0], -1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = svm_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
            
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        return conf_matrix, accuracy, report, class_names
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return None, None, None, None

def plot_to_base64(plt_figure):
    img = io.BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_dataset_stats():
    # Count files per speaker directory
    speaker_counts = {}
    speakers = []
    for speaker_dir in os.listdir(app.config['DATASET_FOLDER']):
        if os.path.isdir(os.path.join(app.config['DATASET_FOLDER'], speaker_dir)):
            speakers.append(speaker_dir)
            count = len(glob.glob(os.path.join(app.config['DATASET_FOLDER'], speaker_dir, "*.wav")))
            speaker_counts[speaker_dir] = count
    
    return speaker_counts, speakers

@app.route('/model_info')
def model_info():
    # Get dataset statistics
    speaker_counts, speakers = get_dataset_stats()
    
    # Create dataset graph
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(speaker_counts.keys(), speaker_counts.values())
    ax1.set_title('Number of Audio Files per Speaker')
    ax1.set_xlabel('Speaker')
    ax1.set_ylabel('File Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    dataset_graph = plot_to_base64(fig1)
    plt.close(fig1)
    
    # Generate confusion matrix and model metrics
    cm, accuracy, report, class_names = generate_confusion_matrix()
    print("important details \n\n\n",cm, accuracy, report, class_names)
    
    confusion_matrix_img = None
    if cm is not None:
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        plt.tight_layout()
        confusion_matrix_img = plot_to_base64(fig2)
        plt.close(fig2)
    
    return render_template(
        'model_info.html', 
        dataset_graph=dataset_graph,
        confusion_matrix=confusion_matrix_img,
        classification_report=report if report else None,
        model_accuracy=accuracy if accuracy else "Model data not available",
        speakers=speakers
    )
data_dir = "Segment_Audio"
model_dir = "Speaker_Models"
sampling_rate = 16000
segment_length = 1  # seconds
n_mfcc = 13

def segment_audio(audio_path, speaker_name, save_path):
    print(audio_path)
    print(speaker_name)
    print(save_path)
    save_path = os.path.join(save_path, speaker_name)
    os.makedirs(save_path, exist_ok=True)
    
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(total_duration // segment_length)
    
    for i in range(num_segments):
        start_sample = i * sampling_rate * segment_length
        end_sample = start_sample + sampling_rate * segment_length
        segment = audio[start_sample:end_sample]
        segment_filename = os.path.join(save_path, f"{i}.wav")
        sf.write(segment_filename, segment, sampling_rate)
    
    print(f"Audio split into {num_segments} segments and saved in {save_path}")

@app.route('/add_speaker', methods=['GET', 'POST'])
def add_speaker():
    if request.method == 'POST':
        speaker_name = request.form.get('speaker_name')
        if not speaker_name:
            return "Speaker name is required", 400
        
        # Create directory for the new speaker if it doesn't exist
        speaker_dir = os.path.join(app.config['DATASET_FOLDER'], speaker_name)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Process audio file
        if 'audio' in request.files and request.files['audio'].filename != '':
            file = request.files['audio']
            filename = f"{speaker_name}.wav"
            file_path = os.path.join(os.getcwd(), filename)
            file.save(file_path)
            print(file_path,"\n\n\n")
            segment_audio(file_path, speaker_name, "Segment_Audio")
        
        # Process recorded audio
        elif 'recordedAudio' in request.form and request.form['recordedAudio']:
            recorded_data = request.form['recordedAudio']
            audio_data = base64.b64decode(recorded_data.split(',')[1])
            
            filename = f"{speaker_name}.wav"
            file_path = os.path.join(os.getcwd(), filename)
            with open(file_path, "wb") as f:
                f.write(audio_data)
            segment_audio(file_path, speaker_name, "Segment_Audio")
            
            
        else:
            return "No audio provided", 400
        
        # Retrain the model
        retrain_model("Segment_Audio", status=False)
        
        return redirect(url_for('model_info'))
    
    # For GET request, render the form with reading passage
    reading_passage = """
    The Rainbow Passage
    
    When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.
    
    When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways. Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain. The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky.
    """
    
    return render_template('add_speaker.html', reading_passage=reading_passage)

if __name__ == '__main__':
    app.run(debug=True)