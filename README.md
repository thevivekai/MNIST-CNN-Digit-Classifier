# Project-1: Convolutional Neural Network for MNIST Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/thevivekai/)

Welcome to my first major project in machine learning\! This repository contains a complete implementation of a Convolutional Neural Network (CNN) designed to classify handwritten digits from the classic MNIST dataset. This project serves as a foundational exercise in computer vision, demonstrating the entire workflow from data preprocessing to model training, evaluation, and analysis. 🚀

## 📜 Table of Contents

  * [Learning Objectives](https://www.google.com/search?q=%23%F0%9F%8E%AF-learning-objectives)
  * [Dataset Information](https://www.google.com/search?q=%23%F0%9F%93%8A-dataset-information)
  * [Methodology](https://www.google.com/search?q=%23-methodology)
  * [CNN Model Architecture](https://www.google.com/search?q=%23%F0%9F%8F%97%EF%B8%8F-cnn-model-architecture)
  * [Performance & Results](https://www.google.com/search?q=%23%F0%9F%93%88-performance--results)
  * [How to Run This Project](https://www.google.com/search?q=%23-how-to-run-this-project)
  * [File Descriptions](https://www.google.com/search?q=%23-file-descriptions)
  * [Conclusion](https://www.google.com/search?q=%23-conclusion)
  * [Contact](https://www.google.com/search?q=%23-contact)

-----

## 🎯 Learning Objectives

This project was undertaken to solidify my understanding and practical skills in the following areas:

  * **CNN Architecture:** Gaining a deep understanding of core components like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
  * **TensorFlow/Keras Implementation:** Building a neural network from scratch using a leading industry framework.
  * **Image Data Preprocessing:** Mastering essential techniques for preparing image data for a classification task, including normalization and reshaping.
  * **Model Evaluation:** Analyzing model performance through various metrics to determine its effectiveness.

-----

## 📊 Dataset Information

The project utilizes the **MNIST dataset**, a cornerstone for learning computer vision.

  * **Content:** It consists of 70,000 grayscale images of handwritten digits (0 through 9).
  * **Training Set:** 60,000 images.
  * **Test Set:** 10,000 images.
  * **Image Dimensions:** Each image is a 28x28 pixel grid.

-----

## 🧠 Methodology

The project follows a systematic machine learning workflow:

1.  **Import Libraries:** Essential libraries like TensorFlow, Keras, NumPy, and Matplotlib were imported.
2.  **Load and Preprocess Data:**
      * The MNIST dataset was loaded directly via `tf.keras.datasets.mnist.load_data()`.
      * **Normalization:** Pixel values were scaled from the original `0-255` range to `0-1` to help the model converge faster.
      * **Reshaping:** A channel dimension was added to the images, changing their shape from `(28, 28)` to `(28, 28, 1)`, which is the required input format for the CNN.
      * **One-Hot Encoding:** The target labels were converted into a categorical format.
3.  **Build CNN Model:** The neural network was constructed layer-by-layer using the Keras Sequential API.
4.  **Compile the Model:** The model was compiled with the `Adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the evaluation metric.
5.  **Train the Model:** The CNN was trained on the preprocessed training data for 20 epochs with a batch size of 128.
6.  **Evaluate Performance:** The trained model was evaluated on the unseen test data to measure its real-world performance.

-----

## 🏗️ CNN Model Architecture

The model architecture is a sequential stack of layers designed to learn hierarchical features from the images.

| Layer Type | Parameters | Output Shape |
| :--- | :--- | :--- |
| `Conv2D` | 32 filters, 3x3 kernel, ReLU, 'same' padding | (None, 28, 28, 32) |
| `MaxPooling2D`| 2x2 pool size | (None, 14, 14, 32) |
| `Conv2D` | 64 filters, 3x3 kernel, ReLU | (None, 12, 12, 64) |
| `MaxPooling2D`| 2x2 pool size | (None, 6, 6, 64) |
| `Conv2D` | 64 filters, 3x3 kernel, ReLU | (None, 4, 4, 64) |
| `Flatten` | - | (None, 1024) |
| `Dense` | 64 units, ReLU | (None, 64) |
| `Dense` | 10 units, Softmax | (None, 10) |

**Total Parameters:** 121,994

Here is a visual representation of the model:

-----

## 📈 Performance & Results

The model achieved outstanding results on the test dataset, demonstrating its effectiveness.

  * **Overall Test Accuracy:** **99.31%**
  * **Overall Test Loss:** **0.035**

### Training History

The training and validation plots show that the model learned effectively without significant overfitting.

### Classification Report

The model performs exceptionally well across all 10 digit classes, with high precision, recall, and F1-scores.

| Class | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| **0** | 0.9939 | 0.9990 | 0.9964 |
| **1** | 0.9973 | 0.9947 | 0.9960 |
| **2** | 0.9942 | 0.9971 | 0.9956 |
| **3** | 0.9844 | 0.9970 | 0.9907 |
| **4** | 0.9919 | 0.9959 | 0.9939 |
| **5** | 0.9888 | 0.9854 | 0.9871 |
| **6** | 0.9989 | 0.9875 | 0.9932 |
| **7** | 0.9941 | 0.9903 | 0.9922 |
| **8** | 0.9949 | 0.9928 | 0.9938 |
| **9** | 0.9921 | 0.9901 | 0.9911 |

### Confusion Matrix

The confusion matrix highlights the model's high accuracy, with the vast majority of predictions falling along the main diagonal (correct classifications).

-----

## 🚀 How to Run This Project

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    ```
2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
    ```
3.  **Run the Jupyter Notebook:**
    Navigate to the project directory and launch the notebook.
    ```bash
    jupyter notebook P1_CNN_MNIST.ipynb
    ```
    You can then run the cells sequentially to see the process and results.

-----

## 📁 File Descriptions

  * **`P1_CNN_MNIST.ipynb`**: The main Jupyter Notebook containing all the Python code, analysis, and outputs.
  * **`P1_CNN_MNIST.html`**: An HTML export of the completed notebook for easy viewing in a browser.
  * **`Assignment Project-1_CNN.pdf`**: The original project description and problem statement.
  * **`model_architecture.png`**: The diagram of the CNN model's architecture.
  * **`README.md`**: This file\!

-----

## ✨ Conclusion

This project successfully demonstrates the power of Convolutional Neural Networks for image classification tasks. The model achieved a high accuracy of **99.31%** on the MNIST test set, proving its ability to generalize well to unseen data. The detailed evaluation confirms robust performance across all digit classes. This exercise has been an excellent practical introduction to building and evaluating deep learning models for computer vision.

-----

## 📧 Contact

**Vivek Prakash Upreti**

  * **Email:** thevivekai@gmail.com
  * **LinkedIn:** [https://linkedin.com/in/thevivekai](https://linkedin.com/in/thevivekai)
