🧠 Fingerprint Detection using Perceptual Image Hashing by CNN (ResNet18 & ResNet50)
This project implements a perceptual image hashing pipeline using deep CNNs to generate compact and discriminative hash vectors from grayscale images. It is primarily tested on the SOCOFing fingerprint dataset.

🚀 Highlights
🔧 Models Used: Modified ResNet18 (Model A) and ResNet50 (Model B) for grayscale input.
🧬 Architecture:
Model A: 7×7×512 → GlobalAvgPool → FC(256 → hash)
Model B: 7×7×2048 → FC(1024 → 512 → hash)
✅ Hash Comparison: Cosine similarity for robust distance calculation.
📦 Dataset: SOCOFing – 6,000+ original and altered fingerprint images.
🛠 Requirements
💻 Hardware
CPU: Intel i7+ or CUDA-enabled GPU
RAM: ≥ 16 GB
Storage: ≥ 500 GB
🧪 Software
Python 3.7+
Libraries: torch, torchvision, numpy, matplotlib, opencv-python
Pre-trained weights: ResNet18 & ResNet50 (ImageNet)
📊 Applications
Image retrieval & matching
Biometric authentication
Tamper detection
🧮 Similarity Metric
We use cosine similarity to compare image hash vectors:

Measures orientation, not magnitude.
Robust to scale and illumination changes.
Ensures images with similar content yield similarity scores close to 1.0.
