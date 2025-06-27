# 🤟 Sign Language Recognition System (MediaPipe + MLPClassifier + OpenCV)

An AI-powered system that recognizes American Sign Language (A–E) in real-time using hand landmarks from MediaPipe and a machine learning classifier. This project helps bridge communication gaps for the hearing and speech impaired.

---

# 🚀 Features

- Real-time hand gesture detection using webcam
- Collect labeled hand landmarks for A–E gestures
- Train an MLPClassifier model on collected hand landmark data
- Predict and display live sign labels in webcam feed
- Supports custom dataset expansion

---

# 🛠️ Tech Stack

- Python
- MediaPipe (for hand tracking)
- OpenCV (for webcam interaction)
- scikit-learn (MLPClassifier)
- NumPy, Pandas (data processing)

---

# 📂 Folder Structure

sign_language_recognition/
├── app.py # (Optional) For future Streamlit app
├── mediapipe_hand_capture.py # Data collection script
├── train_model.py # Model training script
├── predict_webcam.py # Real-time sign prediction
├── predict_sign.py # Predict from saved image or CSV
├── check_labels.py # Check label distribution
├── inspect_data.py # Inspect CSV data quality
├── test_cam.py # Webcam test
├── requirements.txt
├── .gitignore
├── .python-version
├── data/ # Collected hand landmark CSV
│ └── sign_data.csv
├── model/ # Trained model
│ └── sign_model.pkl

---

# 📸 Sample Output

- 🖐️ Hand landmarks detected using MediaPipe
- ✅ Sign A to E predicted in real-time
- 📊 Model accuracy shown after training

---

# 🙋‍♂️ Created By

Mohammed Salman
📧 mohammed.salman.p.2004@gmail.com
🔗 LinkedIn • 🌐 GitHub

---

# 📜 License

This project is licensed under the MIT License.
