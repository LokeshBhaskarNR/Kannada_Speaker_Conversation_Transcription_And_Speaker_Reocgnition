import os
import time
import psutil
import GPUtil
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pydub import AudioSegment
from simple_diarizer.diarizer import Diarizer
from simple_diarizer.utils import check_wav_16khz_mono, convert_wavfile

# ===== SYSTEM STATUS CHECK =====
def system_check():
    print("\n===== SYSTEM STATUS =====")

    # RAM Usage
    ram = psutil.virtual_memory()
    print(f"Available RAM: {ram.available / (1024 ** 3):.2f} GB / {ram.total / (1024 ** 3):.2f} GB")

    # CPU Usage
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_usage}%")

    # GPU Status
    gpus = GPUtil.getGPUs()
    if gpus:
        for gpu in gpus:
            print(f"GPU: {gpu.name}, Memory: {gpu.memoryFree:.2f}MB Free / {gpu.memoryTotal:.2f}MB Total")
    else:
        print("No GPU detected!")

# ===== DIRECTORY HANDLING =====
def ensure_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)

# ===== AUDIO CONVERSION =====
def convert_to_wav(input_file):

    audio = AudioSegment.from_file(input_file)
    output_file = os.path.splitext(input_file)[0] + ".wav"
    audio.export(output_file, format="wav")
    print(f"File converted and saved as {output_file}")
    return output_file 

def convert_audio_if_needed(input_audio):

    converted_audio = "converted_audio.wav"
    if not check_wav_16khz_mono(input_audio):
        return convert_wavfile(input_audio, converted_audio)
    return input_audio

# ===== DIARIZATION FUNCTION =====
def perform_diarization(audio_file, num_speakers):

    signal, fs = sf.read(audio_file)
    diar = Diarizer(embed_model='ecapa', cluster_method='sc', window=1.5, period=0.75)
    start_time = time.time()
    segments = diar.diarize(audio_file, num_speakers=num_speakers)
    end_time = time.time()
    print(f"Diarization completed in {end_time - start_time:.2f} seconds")
    print(segments)
    return signal, fs, segments

# ===== WAVEFORM VISUALIZATION =====
def plot_waveform(signal, fs, title="Waveform of Audio"):

    time_axis = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.figure(figsize=(20, 3))
    plt.plot(time_axis, signal, color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.savefig("images/waveform.png")
    plt.close()

def plot_speaker_segments(signal, fs, segments):

    time_axis = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, signal, color="gray", alpha=0.5, label="Waveform")

    for segment in segments:
        speaker_label = segment.get('spk', segment.get('label', 'Unknown'))  # Use 'label' as fallback
        plt.axvspan(segment['start'], segment['end'], alpha=0.3, label=f"Speaker {speaker_label}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Speaker Diarization Segments")
    plt.legend()
    plt.savefig("images/speaker_segments.png")
    plt.close()

# ===== SAVE SPEAKER SEGMENTS =====
def save_speaker_segments(audio_file, segments, output_dir):

    ensure_directory(output_dir)
    signal, fs = sf.read(audio_file)
    
    timeline_path = "speaker_timeline.txt"

    with open(timeline_path, "w") as timeline_file:

        for segment in segments:
            speaker_id = segment.get('spk', segment.get('label', 'Unknown')) 
            start_sec = segment['start']
            end_sec = segment['end']
            
            start_sample = int(start_sec * fs)
            end_sample = int(end_sec * fs)
            segment_audio = signal[start_sample:end_sample]
            
            segment_filename = os.path.join(
                output_dir, f"speaker_{start_sec:.2f}_to_{end_sec:.2f}.wav"
            )
            sf.write(segment_filename, segment_audio, fs)
            print(f"Saved: {segment_filename}")
            
            timeline_file.write(f"Speaker : {start_sec:.2f}s - {end_sec:.2f}s\n")

    print(f"Speaker timeline saved: {timeline_path}")

# ===== PROCESS AUDIO FILE =====
def process_audio(input_audio, num_speakers, output_dir="Speaker_Split_Audios"):

    wav_audio = convert_to_wav(input_audio)  
    converted_audio = convert_audio_if_needed(wav_audio)  
    
    signal, fs, segments = perform_diarization(converted_audio, num_speakers)
    
    plot_waveform(signal, fs)
    plot_speaker_segments(signal, fs, segments)
    
    save_speaker_segments(converted_audio, segments, output_dir)
    print("Speaker segmentation completed and saved.")

