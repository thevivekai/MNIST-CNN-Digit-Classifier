# CNN for MNIST Digit Recognition

This project demonstrates the implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the well-known MNIST dataset. The goal is to achieve high accuracy in digit recognition.

## Features

*   **Data Loading and Preprocessing:** The MNIST dataset is loaded directly from the Keras library. The images are normalized and reshaped to be suitable for the CNN model.
*   **CNN Model:** A sequential CNN model is built with multiple convolutional, max-pooling, and dense layers.
*   **Training and Evaluation:** The model is trained on the training set and evaluated on the test set to measure its performance.
*   **Visualization:** The training history (accuracy and loss) and the confusion matrix are visualized to assess the model's effectiveness.

## Getting Started

### Prerequisites

To run this project, you need to have Python and the following libraries installed:

*   TensorFlow
*   Keras
*   NumPy
*   Pandas
*   Matplotlib
*   Seaborn
*   Scikit-learn

You can install these dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Usage

1.  Clone this repository to your local machine.
2.  Open the Jupyter Notebook (`P1_CNN_MNIST.ipynb`) to view and run the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.
