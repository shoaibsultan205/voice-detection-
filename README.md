# 🎙️ Real-Time Voice Emotion Detection using Python & TensorFlow

A real-time voice emotion detection system built with **Python**, **TensorFlow**, and **Librosa**.  
This project records your voice, automatically detects silence, and analyzes your **dominant emotion** (like happy, sad, angry, etc.) using a deep learning model.

---

## 🌟 Features
- 🎧 Live voice recording using `sounddevice`
- 🤖 Emotion recognition with a trained TensorFlow `.h5` model
- 🧠 MFCC-based feature extraction using `librosa`
- ⏸️ Auto stop after 5 seconds of silence
- 📊 Displays dominant and distributed emotions
- ⚡ Lightweight and real-time performance

---

## 🧩 Supported Emotions
| ID | Emotion |
|----|----------|
| 1 | Neutral |
| 2 | Calm |
| 3 | Happy |
| 4 | Sad |
| 5 | Angry |
| 6 | Fearful |
| 7 | Disgust |
| 8 | Surprised |

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/voice-emotion-detection.git
cd voice-emotion-detection
