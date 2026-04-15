🧠 Low-Bias Face Recognition System

![GitHub repo size](https://img.shields.io/github/repo-size/probal2005/face_recognition_project)
![GitHub stars](https://img.shields.io/github/stars/probal2005/face_recognition_project?style=social)
![GitHub forks](https://img.shields.io/github/forks/probal2005/face_recognition_project?style=social)
![GitHub issues](https://img.shields.io/github/issues/probal2005/face_recognition_project)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

![Status](https://img.shields.io/badge/Status-Active-success)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)


🧠 Low-Bias Face Recognition System

A fairness-aware face recognition system built using PyTorch that reduces demographic bias using ArcFace loss and MMD (Maximum Mean Discrepancy) regularization.
This project demonstrates how to design a machine learning pipeline that not only optimizes accuracy but also ensures equitable performance across demographic groups.

🚀 Features

🔍 Face Recognition Model
Lightweight CNN backbone
Embedding-based recognition
ArcFace loss for improved class separation

⚖️ Fairness-Aware Training
Uses MMD (Maximum Mean Discrepancy) to reduce bias
Minimizes feature distribution differences between groups (e.g., gender)

📊 Evaluation Metrics
Male vs Female accuracy
Accuracy gap (fairness metric)
Overall accuracy

📈 Visualization
Training loss curves
Accuracy comparison graphs
Fairness gap tracking

🧪 Synthetic Dataset
Lightweight dataset for fast experimentation
No external dataset required

🏗️ Project Structure
.
├── data/                  # Dataset directory (auto-created)
├── models/                # Saved models
├── results/               # Output results
├── training_results.png   # Visualization output
├── fair_face_model.pth   # Trained model
└── main.py               # Main training script


⚙️ Installation
Install dependencies:
pip install torch torchvision numpy matplotlib tqdm scikit-learn


▶️ Usage
Run the training script:
python main.py


🧪 Training Details
Parameter
Value
Epochs
15
Batch Size
8
Embedding Size
64
Optimizer
SGD
Learning Rate
0.01
Fairness Weight
0.01


🧠 Model Architecture
Backbone
Conv → BatchNorm → ReLU layers
Adaptive pooling
Feature Extractor
Fully connected embedding layer
Loss Functions
ArcFace Loss (classification)
MMD Loss (fairness regularization)

⚖️ Fairness Approach
The system reduces bias by enforcing similarity between feature distributions of demographic groups.
MMD Loss Objective:
Align feature distributions across groups (e.g., male vs female)
Reduce performance disparity
Total Loss:
Total Loss = ArcFace Loss + λ × Fairness Loss


📊 Evaluation Metrics
Male Accuracy
Female Accuracy
Accuracy Gap
Overall Accuracy
✅ Goal: Keep accuracy gap < 5%

📈 Output Example
After training, the system generates:
📉 Loss curves
📊 Accuracy per group
⚖️ Fairness gap plot
📁 Saved model (.pth)

💡 Notes
Designed for CPU-friendly execution
Uses synthetic data for demonstration
Easily extendable to real datasets like:
CelebA
FairFace
VGGFace2

🔧 Customization
You can modify:
lambda_fair = 0.01   # fairness strength
num_samples = 300    # dataset size
num_epochs = 15      # training duration


⚠️ Limitations
Synthetic dataset → not real-world performance
Binary gender only (for demo)
Limited demographic attributes

🔮 Future Improvements
Use real-world datasets
Add race/age fairness
Improve backbone (ResNet, MobileNet)
Deploy as API or web app

🤝 Contributing
Contributions are welcome! Feel free to:
Open issues
Suggest improvements
Submit pull requests

📜 License
This project is open-source and available under the MIT License.

👨‍💻 Author
Probal Dhali
