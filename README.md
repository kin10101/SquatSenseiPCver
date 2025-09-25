# SquatSensei PC Version 🏋️‍♂️

*AI-Powered Real-Time Squat Form Analyzer*

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-red)

## 🚀 Overview

SquatSensei is an intelligent fitness application that uses computer vision and machine learning to analyze squat form in real-time. By leveraging MediaPipe pose estimation and custom-trained models, it provides instant feedback on squat technique, counts repetitions automatically, and helps users improve their form through audio feedback.

### Key Highlights
- **Real-time pose analysis** using MediaPipe and OpenCV
- **Multi-aspect form checking**: stance, head position, chest position, knee alignment, and heel placement
- **Automatic rep counting** with phase detection (top → middle → bottom → middle → top)
- **Audio feedback system** with binary error codes for different form issues
- **Data collection tools** for training custom models
- **GUI interface** built with Tkinter for easy interaction

## ✨ Features

### Core Functionality
- 🎯 **Real-Time Squat Analysis**: Live camera feed with pose detection overlay
- 📊 **Form Assessment**: Analyzes 5 key aspects of squat form
  - Stance width (wide/standard/narrow)
  - Head position (forward/downward facing)
  - Chest position (up/down)
  - Knee alignment (correct/collapsing inward)
  - Heel placement (flat/raising)
- 🔢 **Automatic Rep Counting**: Tracks complete squat cycles
- 🔊 **Audio Feedback**: Provides specific feedback based on detected errors
- 📹 **Video Processing**: Can analyze pre-recorded videos
- 🎛️ **Calibration System**: Adjustable for different users

### Machine Learning Models
- **Random Forest Classifiers** for each form aspect
- **LSTM Models** for sequence-based analysis
- **MLP Models** for alternative classification approaches
- **Phase Detection** using hip-to-knee ratio analysis

### Data Collection & Training
- 📊 **Dataset Generation**: Tools to create training data from videos
- 🎥 **Video Processing**: Automatic landmark extraction and labeling
- 📈 **Model Training**: Scripts for training custom classifiers
- 🔍 **Data Visualization**: Tools to analyze dataset distributions

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or IP camera for live analysis
- Windows/Linux/macOS compatible

### Required Dependencies
```bash
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install pandas
pip install numpy
pip install keras
pip install tensorflow
pip install playsound
pip install pillow
pip install yt-dlp  # For YouTube video downloads
```

### Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SquatSenseiPCver.git
   cd SquatSenseiPCver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # You may need to create this file
   ```

3. **Create necessary directories**
   ```bash
   mkdir Models  # For storing trained models
   mkdir Datasets  # For training data
   mkdir Feedback  # For audio feedback files
   ```

4. **Download or train models**
   - Run the model training scripts or download pre-trained models
   - Place `.pkl` model files in the `Models/` directory

5. **Set up audio feedback**
   - Ensure audio feedback files are in the `Feedback/` directory
   - Files should be named with binary error codes (e.g., `0000.mp3`, `1111.mp3`)

## 🎮 Usage

### Running SquatSensei
```bash
python SQUATSENSEI.py
```

### GUI Interface
1. **Start Live Classification**: Begin real-time squat analysis
2. **Calibrate**: Adjust settings for your setup
3. **Exit**: Close the application

### Command Line Alternatives
- **Test with pre-trained models**: `python test_model.py`
- **Keras model testing**: `python keras_test_model.py`
- **Data collection**: `python data_collector_video.py`

### Understanding the Feedback
The system provides visual and audio feedback:
- **Visual**: Color-coded pose landmarks and text overlays
  - Green: Top position
  - Yellow: Middle position
  - Red: Bottom position
- **Audio**: Binary-coded feedback based on detected errors
  - Each aspect (head, chest, knee, heel) contributes to a 4-bit error code
  - `0` indicates error, `1` indicates correct form

## 📁 Project Structure

```
SquatSenseiPCver/
├── SQUATSENSEI.py          # Main GUI application
├── model.py                # Random Forest model training
├── lstm_model.py           # LSTM model implementation
├── mlp_model.py           # Multi-layer perceptron model
├── keras_test_model.py    # Keras model testing
├── test_model.py          # Model testing and evaluation
├── data_collector_video.py # Video data collection tool
├── dataset_visualizer.py  # Dataset analysis and visualization
├── rep_seperator.py       # Video segmentation tool
│
├── Datasets/              # Training data
│   ├── kel.csv           # Kelvin's training data
│   ├── kin.csv           # Kin's training data
│   └── etc.csv           # Additional training data
│
├── Feedback/              # Audio feedback files
│   ├── 0000.mp3          # Perfect form
│   ├── 0001.mp3          # Heel error only
│   └── ...               # Other error combinations
│
├── Utilities/             # Helper tools
│   ├── csv_header.py     # CSV structure setup
│   ├── folder_sorter.py  # File organization
│   ├── video_compressor.py # Video processing
│   └── yt_downloader.py   # YouTube video downloader
│
└── Models/                # Trained model files (not in repo)
    ├── stance_model.pkl
    ├── phase_model.pkl
    ├── head_model.pkl
    ├── chest_model.pkl
    ├── knee_model.pkl
    └── heel_model.pkl
```

## 🔬 Technical Details

### Architecture
The system follows a modular architecture with clear separation of concerns:

1. **Pose Detection**: MediaPipe extracts 33 body landmarks
2. **Feature Engineering**: Relevant landmarks selected for each classification task
3. **Classification**: Separate models for each form aspect
4. **Phase Detection**: Hip-to-knee ratio determines squat phase
5. **Rep Counting**: State machine tracks complete squat cycles
6. **Feedback Generation**: Error codes mapped to audio files

### Machine Learning Approach
- **Random Forest**: Primary classification algorithm for robustness
- **Feature Selection**: Specific landmark subsets for each classification task
- **Data Preprocessing**: Landmark normalization and phase filtering
- **Multi-model Ensemble**: Separate specialized models for each form aspect

### Data Flow
```
Camera Input → MediaPipe → Landmark Extraction → Feature Engineering → 
ML Classification → Form Analysis → Rep Counting → Feedback Generation
```

## 🔧 Model Training

### Creating Training Data
1. Use `data_collector_video.py` to process videos and extract landmarks
2. Manually label form aspects for each frame
3. Save data to CSV files in the `Datasets/` directory

### Training Models
```bash
python model.py          # Train Random Forest models
python lstm_model.py     # Train LSTM models
python mlp_model.py      # Train MLP models
```

### Model Evaluation
```bash
python test_model.py     # Evaluate model performance
```

## 🎯 Error Code System

The audio feedback system uses a 4-bit binary code where each bit represents a form aspect:
- **Bit 1**: Head position (0=error, 1=correct)
- **Bit 2**: Chest position (0=error, 1=correct)  
- **Bit 3**: Knee alignment (0=error, 1=correct)
- **Bit 4**: Heel placement (0=error, 1=correct)

Examples:
- `1111.mp3`: Perfect form
- `0111.mp3`: Head position error only
- `0000.mp3`: All aspects incorrect

## 🛠️ Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera permissions and connectivity
2. **Models not loading**: Ensure model files are in the `Models/` directory
3. **Audio not playing**: Check audio file paths and system audio settings
4. **Poor detection accuracy**: Ensure good lighting and clear camera view

### Performance Optimization
- Use adequate lighting for better pose detection
- Ensure the full body is visible in the camera frame
- Maintain consistent distance from the camera
- Calibrate the system for your specific setup

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

### Areas for Contribution
- Additional form aspects (e.g., depth analysis)
- Mobile app version
- Web interface
- More sophisticated ML models
- Additional exercise types

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** team for the excellent pose estimation framework
- **OpenCV** community for computer vision tools
- **Scikit-learn** developers for machine learning utilities
- All contributors who helped with data collection and testing

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the repository issues or contact the maintainers directly.

---

*Built with ❤️ for the fitness community*