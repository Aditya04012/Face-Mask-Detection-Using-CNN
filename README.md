# 😷 Face Mask Detection using CNN (Accuracy: 98.90%)

This project detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) model, real-time webcam feed, and Mediapipe's face detection.

## 📌 Overview

Face Mask Detection has become crucial in the context of global health. This real-time application:

- Detects faces using Mediapipe
- Classifies the face as:
  - **With Mask** ✅
  - **Without Mask** ❌
- Utilizes a trained CNN model with **98.90%** validation accuracy.

## 🧠 Model

- **Architecture**: Custom CNN built using TensorFlow/Keras
- **Input Size**: 128x128 RGB images
- **Output**: Binary classification (with_mask / without_mask)
- **Accuracy**: 98.90% on validation data

Model file: `Face_mask_98.90%.h5`

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **Mediapipe**
- **NumPy**

## 🖼️ Sample Results

Real-time classification from webcam:

- 🟩 Green box: **With Mask**
- 🟥 Red box: **Without Mask**

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya04012/Face-Mask-Detection-Using-CNN
cd Face-Mask-Detection-Using-CNN
```

### 2. Install Dependencies

Make sure you have Python installed (>=3.7), then:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```txt
opencv-python
tensorflow
mediapipe
numpy
```

### 3. Add Trained Model

Place the file `Face_mask_98.90%.h5` in the root directory of the project.

### 4. Run the Script

```bash
python detect_mask.py
```

Press **`q`** to quit the webcam window.

## 📂 File Structure

```
Face-Mask-Detection-Using-CNN/
│
├── Face_mask_98.90%.h5         # Trained CNN model
├── main.py                     # Main Python script
├── Notebook.ipynb              # Notebook            
├── README.md                   # Project description
└── requirements.txt            # Python dependencies
```

## 📈 Accuracy

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 98.90%    |
| Model Type   | CNN       |
| Image Size   | 128x128   |

## 🙋‍♂️ Author

**Aditya Bhatnagar**  
📧 adityabhatnagar0403@gmail.com  
🔗 [GitHub](https://github.com/Aditya04012)