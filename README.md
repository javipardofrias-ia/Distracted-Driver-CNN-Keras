
# Distracted-Driver-CNN-Keras — CNN for Distracted Driver Detection

## 📘 Description

This project implements a Convolutional Neural Network (CNN) using the `Keras` library (part of `TensorFlow`) to classify images of drivers. The dataset, `state-farm-distracted-driver-detection`, contains images showing various driving behaviors, and the model's goal is to accurately identify these behaviors to detect distracted driving. The notebook provides a comprehensive guide through the entire machine learning pipeline for a multi-class image classification task.

The project demonstrates key concepts such as data preprocessing, which includes handling a class imbalance, and defining a robust CNN architecture with multiple convolutional and dense layers for effective feature extraction and classification.

## 🛠️ Requirements

Ensure you have Python and the following libraries installed:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

Libraries used:

  * `tensorflow` & `keras` – The primary libraries for building, compiling, and training the CNN model.
  * `numpy` – For numerical operations.
  * `pandas` – For data manipulation and CSV file handling.
  * `scikit-learn` – For dataset splitting (`train_test_split`).
  * `matplotlib.pyplot` – For data visualization.

## 📊 Data Preprocessing

Preprocessing is a critical step to prepare the image data for the CNN. This project follows these steps:

1.  **Dataset Import**: The `state-farm-distracted-driver-detection` dataset is imported as a zip file and uncompressed.
2.  **Dimensionality Reduction**: The dataset is intentionally reduced by 25% by sampling the `driver_imgs_list.csv` file and removing the corresponding image files from the disk. This step reduces the overall size of the dataset for faster experimentation.
3.  **Class Balancing**: The dataset is balanced using oversampling to ensure each of the 10 classes has the same number of images as the largest class. This is a crucial step to prevent the model from becoming biased towards the majority classes.
4.  **Data Splitting**: The dataset is split into training, validation, and test sets. The training data is split again to create a validation set.
5.  **Image Generation and Normalization**: `ImageDataGenerator` is used to create batches of images. This tool also rescales the pixel values from the `[0, 255]` range to `[0, 1]` and resizes all images to `224x224` pixels, which are essential steps for neural network training.

## 🧠 AI Model Structure

The neural network is a CNN implemented using the Keras Functional API. The architecture is designed to effectively extract features from images.

  * **Input Layer**: The input shape is defined as `224x224` pixels with 3 channels (RGB).
  * **Convolutional Blocks**: The model includes a series of four `Conv2D` layers with increasing filter sizes (512, 256, 128, and 64), each followed by a `ReLU` activation function and a `MaxPooling2D` layer. This structure helps the model learn hierarchical features from the image data.
  * **Flattening and Regularization**: The output from the convolutional blocks is flattened into a 1D vector using `Flatten`. A `Dropout` layer with a rate of `0.2` is applied to prevent overfitting.
  * **Dense Layers**: A hidden `Dense` layer with 600 neurons and `ReLU` activation is used, followed by another `Dropout` layer of `0.2`.
  * **Output Layer**: The final `Dense` layer has 10 neurons, corresponding to the 10 driver classes, and uses a `softmax` activation function to output a probability distribution over the classes.

## 🧬 Training and Results

The training process is configured to optimize the model's performance on the classification task.

  * **Loss Function**: `sparse_categorical_crossentropy` is used to measure the difference between the model's predictions and the true labels, which is appropriate for a multi-class classification problem with integer labels.
  * **Optimization**: The model is compiled with the `Adam` optimizer.
  * **Training and Evaluation**: The model is trained using the prepared data generators. An `EarlyStopping` callback is used to stop the training if the model's performance no longer improves, as per the notebook's instructions. The performance is evaluated by tracking the loss and accuracy on both the training and validation sets.

## 🚀 How to Run

1.  Ensure the `state-farm-distracted-driver-detection.zip` file is available in your Google Drive.
2.  Open the notebook:
    ```bash
    jupyter notebook L3P1_ClasificarImagenes.ipynb
    ```
3.  Run the cells sequentially to:
      * Mount Google Drive and import the dataset.
      * Preprocess the data (reduce dimensionality, balance classes, and split).
      * Define, compile, and train the CNN model.
      * Evaluate the model's performance.
