# Image Classifier using Fashion MNIST Dataset

## Project Overview
This project focuses on practicing machine learning algorithms by building an image classifier using the **Fashion MNIST** dataset available in the **TensorFlow Keras** library. The Fashion MNIST dataset contains grayscale images of various clothing items such as shirts, shoes, and bags, which are classified into different categories. The project includes data preprocessing, model training, and evaluation using various machine learning algorithms.

## Features
- **Fashion MNIST Dataset**: Used for training and testing the image classification model.
- **Machine Learning Algorithms**: Implemented various machine learning techniques to classify clothing items.
- **Visualization**: Matplotlib is used for visualizing data samples and model performance.
- **Feature Extraction**: Used skimage and OpenCV for feature extraction techniques.
- **Evaluation**: Model performance is evaluated using metrics from `sklearn`.

## Repository Structure
```
.
├── Image Classifier.ipynb  # Jupyter Notebook for model building and experimentation
├── README.md               # Project documentation
```

## Getting Started

### Prerequisites
Ensure you have Python installed. You can install the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Libraries Used
- **Python 3.8+**
- **TensorFlow/Keras**: For loading the Fashion MNIST dataset and building the image classification model.
- **NumPy**: For handling array-based operations.
- **Matplotlib**: For data visualization and plotting model performance.
- **OpenCV (cv2)**: For image manipulation and preprocessing.
- **scikit-image (skimage.feature)**: For extracting image features.
- **scikit-learn (sklearn)**: For model evaluation and accuracy metrics.

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/rHarsh-11/Image_classifier.git
2. Navigate to the project directory:
   ```bash
   cd Image_classifier
   ```
3. Run the Jupyter Notebook for model training and testing:
   ```bash
   jupyter notebook
   ```

### Dataset
The **Fashion MNIST** dataset is preloaded from TensorFlow’s dataset module. It contains 70,000 grayscale images in 10 classes, with 60,000 images for training and 10,000 for testing. Each image is 28x28 pixels in size.

**Classes in the Dataset:**
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Project Details

### Image Preprocessing
- Resized images to ensure uniform input size.
- Normalized pixel values to a range of 0 to 1 for better model performance.
- Used OpenCV (`cv2`) for image manipulation.
- Applied feature extraction techniques using `skimage.feature` to improve model accuracy.

### Model Training
- The classification model was built using **TensorFlow/Keras**. 
- Various algorithms, such as Neural Networks, were implemented and compared for accuracy.
- The model was trained using categorical cross-entropy loss and evaluated on test data.

### Visualization
- Sample images from the dataset were visualized using Matplotlib.
- Loss and accuracy curves were plotted to observe model performance.

### Evaluation
- Used classification accuracy and confusion matrix to assess model performance.
- Fine-tuned the model by adjusting hyperparameters and data augmentation techniques.

## Results & Insights
- The model achieved a high accuracy in classifying clothing items across different categories.
- Key challenges included distinguishing between similar categories (e.g., shirts vs. t-shirts), which could be improved with advanced feature extraction techniques.

## Conclusion
This project demonstrates the effective use of machine learning algorithms for image classification on the Fashion MNIST dataset. It highlights various techniques for improving classification accuracy through preprocessing, feature extraction, and model evaluation.

## Acknowledgements
- Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) by Zalando, available in TensorFlow datasets.
