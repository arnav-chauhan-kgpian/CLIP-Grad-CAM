# CLIP + Grad-ECLIP Visual Explanation

This codebase provides a **simple and practical workflow to interpret the decision-making process of the CLIP (Contrastive Language–Image Pre-training) model** using **Grad-ECLIP (Gradient-based Visual Explanation for CLIP)**.

It was implemented for a **coding task** and focuses on:

* Running zero-shot CLIP inference on video frames
* Generating **saliency heatmaps** that highlight which image regions influence CLIP’s image–text matching

---

## 📌 Overview

The notebook walks through the following key steps:

### 1. Environment Setup

* Installation of essential deep learning and vision libraries
* Core frameworks include **PyTorch** and **OpenMMLab** components

### 2. Video Processing

* Uses the **decord** library for efficient video decoding
* Extracts a specific RGB frame (typically the middle frame) from a video file

### 3. CLIP Inference

* Loads a pretrained CLIP model (e.g., `ViT-B/32`)
* Performs a forward pass to compute **image–text similarity** for a given prompt

### 4. Visual Explanation with Grad-ECLIP

* Applies **Grad-ECLIP** to backpropagate gradients
* Produces **heatmaps** that visualize which regions of the image most influence CLIP’s prediction

---

## 🧰 Prerequisites

To run this, you will need:

* **Google Colab** (recommended) or a **local machine with GPU support**
* Python 3.8+

### Major Dependencies

* **OpenAI CLIP**
* **OpenMMLab stack**: `mmcv`, `mmengine`, `mmagic`
* **decord** – video reading
* `opencv-python-headless`

---

## ⚙️ Setup & Usage

### 1. Clone Required Repositories

```bash
git clone https://github.com/open-mmlab/mmcv.git
git clone https://github.com/Cyang-Zhao/Grad-Eclip
```

### 2. Install Dependencies

* Run the installation cells in the notebook
* This sets up the OpenMMLab environment and other required utilities

### 3. Add Video Data

* Upload your target video file (e.g., `010-cl-02-090.avi`)
* Place it in a directory such as:

  ```
  /content/drive/MyDrive/test_run/
  ```

### 4. Run Inference & Visualization

* Extract the middle frame from the video
* Perform CLIP inference with a text prompt
* Generate the **Grad-ECLIP heatmap**

---

## 📊 Results

The output consists of:

* The **original video frame**
* A **Grad-ECLIP heatmap overlay**

This visual comparison clearly shows **where CLIP focuses** when aligning the image with a given textual description, providing interpretability into CLIP’s multimodal reasoning.

---
* Failure cases
* Comparison with Grad-CAM
