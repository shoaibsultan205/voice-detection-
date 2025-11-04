import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from collections import Counter
import time

# Load your model
model = load_model(r"C:\Users\PC\Downloads\emotion_model (1).h5")

# Emotion labels
emotion_labels = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# Settings
sr = 22050
chunk_duration = 2       # analyze every 2 seconds
silence_threshold = 0.02 # below this volume → consider silence
silence_limit = 5        # stop after 5 sec of silence

print("🎙️ Start speaking... (Recording will stop automatically after you stay silent for 5 seconds.)")

recorded_audio = []
silence_time = 0
start_time = time.time()

while True:
    audio_chunk = sd.rec(int(chunk_duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    chunk_volume = np.abs(audio_chunk).mean()

    if chunk_volume < silence_threshold:
        silence_time += chunk_duration
    else:
        silence_time = 0  # reset if user speaks again
    recorded_audio.extend(audio_chunk.flatten())

    if silence_time >= silence_limit:
        print("🛑 Silence detected — stopping recording.")
        break

print(f"✅ Recording finished. Total duration: {round(time.time() - start_time, 2)} seconds")

# Break into smaller chunks for emotion analysis
audio = np.array(recorded_audio)
samples_per_chunk = int(sr * chunk_duration)
chunks = [audio[i:i + samples_per_chunk] for i in range(0, len(audio), samples_per_chunk)]

def extract_features(y):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return np.expand_dims(mfccs, axis=0)

# Predict emotion per chunk
predictions = []
for chunk in chunks:
    if len(chunk) < samples_per_chunk:
        continue
    features = extract_features(chunk)
    pred = model.predict(features, verbose=0)
    predicted_class = np.argmax(pred)
    predictions.append(predicted_class + 1)

# Final emotion summary
emotion_counts = Counter(predictions)
final_emotion = emotion_labels[emotion_counts.most_common(1)[0][0]]

print("\n🎧 Final Emotion Analysis:")
print(f"Dominant Emotion: {final_emotion}")
print("Emotion Distribution:", {emotion_labels[k]: v for k, v in emotion_counts.items()})
