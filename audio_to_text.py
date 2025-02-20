import os
import speech_recognition as sr

audio_folder_path = "Speaker_Split_Audios"

def transcribe_audio_folder(audio_folder):
    recognizer = sr.Recognizer()
    output_file = "audio_transcriptions.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        for audio_file in sorted(os.listdir(audio_folder)):
            if audio_file.endswith(".wav"):  
                audio_path = os.path.join(audio_folder, audio_file)

                with sr.AudioFile(audio_path) as source:
                    print(f"🎧 Processing: {audio_file}")
                    audio = recognizer.record(source)

                try:
                    text_kn = recognizer.recognize_google(audio, language="kn-IN")

                    file.write(f"{text_kn}\n")

                except sr.UnknownValueError:
                    print(f"❌ Could not understand {audio_file}")
                    file.write(f"🎵 {audio_file}:[Unrecognized Audio]\n")
                except sr.RequestError:
                    print("❌ Network error. Please check your internet connection.")
                    break

    print(f"✅ Transcriptions saved in '{output_file}'")

