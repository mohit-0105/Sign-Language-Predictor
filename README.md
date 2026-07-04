# AI-Powered Sign Language Interpreter ✋🤖

Real-time hand sign recognition using computer vision. A webcam feed is processed with **MediaPipe Hands** to extract 21 hand landmarks, which are fed into a **Random Forest classifier** to recognize signs and display them as live captions.

## Demo

(https://drive.google.com/drive/folders/1qj9gSphQle5GNfS7AXVWLsr1JuweDBNR?usp=sharing)

## Features

- 🖐️ Real-time hand landmark tracking via webcam
- 🧠 Machine learning classifier trained on custom hand-sign data
- 💬 Live on-screen captioning of recognized signs
- 🌍 (Planned) Multi-language translation of recognized signs

## Tech Stack

- **Python 3.11**
- **OpenCV** – webcam capture and frame processing
- **MediaPipe Hands** – hand landmark detection
- **scikit-learn (Random Forest)** – gesture classification

## Project Structure

```
sign-language-project/
├── hand-tracking.py       # Phase 1: webcam + landmark visualization
├── collect-data.py        # Phase 2: record labeled hand landmark data
├── train-model.py         # Phase 3: train Random Forest classifier
├── predict.py             # Phase 4: real-time sign prediction
├── hand_landmarks.csv      # Collected training data (generated)
├── sign_model.pkl          # Trained model (generated)
└── README.md
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/sign-language-project.git
   cd sign-language-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate      # Mac/Linux
   .venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install opencv-python mediapipe pandas scikit-learn joblib
   ```

## Usage

### 1. Test hand tracking
```bash
python hand-tracking.py
```
Confirms your webcam and MediaPipe are working — you should see 21 landmarks tracked on your hand.

### 2. Collect training data
```bash
python collect-data.py
```
Records labeled hand landmark samples for each sign in your list. Press **SPACE** to start recording a sign, **s** to skip, **q** to quit.

### 3. Train the model
```bash
python train-model.py
```
Trains a Random Forest classifier on your collected data and saves it as `sign_model.pkl`. Prints accuracy and a classification report.

### 4. Run real-time prediction
```bash
python predict.py
```
Opens your webcam and displays the predicted sign as a live caption on screen once detection is stable.

## How It Works

1. **Landmark extraction** – MediaPipe detects 21 (x, y, z) hand landmarks per frame.
2. **Data collection** – Landmarks are labeled with the sign being performed and saved to CSV.
3. **Training** – A Random Forest learns to map landmark patterns to sign labels.
4. **Prediction** – Live landmarks are classified frame-by-frame, with a stability buffer to avoid flickery predictions.
5. **Output** – The recognized sign is displayed on screen as a live caption.

## Roadmap

- [x] Real-time hand tracking
- [x] Custom sign data collection pipeline
- [x] Random Forest classifier
- [x] Real-time prediction with stability smoothing
- [ ] Multi-language translation of recognized signs
- [ ] Text-to-speech output (future addition)
- [ ] Expand to full ASL alphabet
- [ ] Two-hand sign support
- [ ] CNN-based model for improved accuracy

## Why This Project

Built as a practical computer vision project demonstrating real-time gesture recognition, with genuine accessibility applications — helping bridge communication between sign language users and non-signers.

## License

MIT License — feel free to use and build on this project.
