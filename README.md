# Image Classifier using CNN

## CODETECH Internship - Task 3

This project implements a Convolutional Neural Network (CNN) for handwritten digit image classification using the scikit-learn digits dataset.

### Workflow
- Load and normalize image data
- Build a CNN with convolution, pooling, dense and dropout layers
- Train with Adam optimization
- Evaluate accuracy and loss
- Generate a classification report and confusion matrix
- Save training curves, predictions and the trained Keras model

### Run
```bash
python -m pip install -r requirements.txt
python image_classifier_cnn.py
```

The generated results are saved in the `outputs` folder.

### Dataset
The educational `sklearn.datasets.load_digits` dataset contains 8x8 grayscale handwritten-digit images for classes 0 through 9.
