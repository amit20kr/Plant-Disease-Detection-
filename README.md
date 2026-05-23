#  AgriVision CBM: Explainable Edge AI for Plant Diseases

---

##  The Problem
Agricultural AI models frequently fail in real-world environments due to three critical flaws:
1. **Domain Bias:** Models memorize laboratory backgrounds (like the grey tables in the PlantVillage dataset) instead of learning actual plant biology.
2. **The Black Box:** Farmers cannot trust a diagnosis they cannot understand.
3. **Cloud Dependency:** Heavy, server-based models are completely useless in rural fields without internet access.

##  Our Solution (The 3-Track Architecture)
We engineered an offline, highly compressed, and fully explainable diagnostic engine using a **Residual Concept Bottleneck Model (CBM)** trained on 15 distinct classes of agricultural diseases.

###  Track A: Robustness via Adversarial Augmentation
To combat background bias, we implemented aggressive Spatial Sabotage (`CoarseDropout`, `MotionBlur`) and L2 Regularization during training. By intentionally obscuring the training data, we forced the neural network to ignore the background and converge purely on the biological lesions.

###  Track B: Explainable AI (XAI)
Instead of a traditional black-box output, our model predicts 4 intermediate agronomic concepts (`Is Healthy`, `Has Spots`, `Is Blighted`, `Complex Pathogen`) *before* making a final diagnosis. 
* **Visual Proof:** We utilized feature-extraction to generate dynamic Grad-CAM heatmaps, proving the AI focuses strictly on necrotic leaf tissue and actively ignores real-world disruptive backgrounds and watermarks.

###  Track C: Lightweight Edge Deployment
Using Post-Training INT8 Quantization via TensorFlow Lite, we compressed our 2.5 Million parameter MobileNetV3 backbone to an ultra-lightweight footprint. 
* **Original FP32 Model:** 7.34 MB
* **Deployed INT8 Edge Model:** **0.78 MB** *(9.4x Compression)*

---

##  Model Performance Metrics
* **Training Accuracy (Feature Convergence):** 88.8%
* **Parameters:** 2.5 Million 

---

##  Repository Structure & How to Run
This repository contains our complete end-to-end pipeline:

1. **`PlantDisease_Training.ipynb`**: The Google Colab notebook detailing our data engineering, undersampling (200 images/class), and CBM architecture training.
2. **`PlantDisease_Inference.ipynb`**: The inference notebook used to test custom real-world edge cases and generate our 3-Panel Executive Heatmap Dashboards.


