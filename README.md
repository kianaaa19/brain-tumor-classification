# Brain Tumor Detection Using MRI Images

## Overview
This project focuses on developing a machine learning model for detecting brain tumors in MRI images. By utilizing deep learning techniques and transfer learning with the DenseNet121 architecture, the aim is to accurately classify images based on the presence of different types of tumors. The categories include glioma, pituitary tumors, meningioma, and images with no tumors. 

## Project Understanding
The objective of this project is to build a robust classifier that can differentiate MRI images of brain tumors from those without tumors. This is particularly important for early detection and diagnosis in medical settings. The project involves multiple stages: data preprocessing, exploratory data analysis (EDA), model development, training, evaluation, and ultimately testing the model using unseen data.

## Data Understanding
The dataset consists of MRI images organized into directories based on tumor type, with a total of approximately 7,023 images, divided into:
- **Training set**: 5,712 images
- **Testing set**: 1,311 images

The categories include:
- Glioma
- Meningioma
- Pituitary tumors
- No tumor

Each image is labeled according to the type of brain tumor (if present), allowing for supervised learning.

## Data Processing
Data processing steps are essential to prepare the images for the model:
1. **Loading Images**: Images are loaded from their respective directories, and paths are stored in a DataFrame alongside their class labels.
2. **Data Augmentation**: To improve model generalization and performance, data augmentation techniques such as brightness adjustments are applied.
3. **Image Resizing**: All images are resized to a uniform shape (224x224 pixels), which is necessary for input to the neural network.
4. **Creating Generators**: The Keras `ImageDataGenerator` is used to create data generators for training, validation, and testing datasets, normalizing image pixel values.

## Exploratory Data Analysis (EDA)
During EDA, various analyses are performed to gain insights into the dataset:
- **Class Distribution**: Visualizations are created to show the counts of images in each tumor class, which helps to identify any imbalances in the dataset.
- **Visualization of Sample Images**: Random samples from each class can be displayed to understand the nature and quality of input data.

These exploratory steps provide an initial understanding of the dataset's characteristics and help identify potential issues before model training.

## Modeling
For the modeling phase, the project employs a deep learning approach using the DenseNet121 architecture:
- **Transfer Learning**: DenseNet121 is chosen for its efficiency in handling complex visual tasks, relying on pre-trained weights from the ImageNet dataset.
- **Model Architecture**: The architecture is adjusted by adding custom layers (such as dropout and fully connected layers) tailored to our specific classification task.
- **Compilation**: The model is compiled using the Adamax optimizer and a categorical crossentropy loss function, which is suitable for multi-class classification tasks.

Once the model is built, it is trained using the training dataset while evaluating its performance on the validation set.

## Evaluation
Model evaluation is critical for assessing performance:
1. **Training Metrics**: Metrics such as accuracy, precision, and recall are tracked during training.
2. **Validation Metrics**: Validation accuracy and loss are calculated to ensure the model is not overfitting the training data.
3. **Confusion Matrix**: A confusion matrix is created to visualize the model's prediction performance across different classes, providing insights into misclassifications.

In this project, the model's test accuracy reached approximately 83%, demonstrating effective performance in distinguishing between tumor types.

## Conclusion
The project successfully demonstrates the application of deep learning techniques for brain tumor detection using MRI images. It leverages a well-structured dataset and advanced methodologies such as transfer learning with dense neural networks. The results indicate that the model can effectively classify images into tumor categories, providing a foundation for future enhancements. Potential improvements could involve optimizing model hyperparameters, including more complex architectures, or expanding the dataset to enhance classification accuracy. Ultimately, this work can contribute significantly to diagnostic processes in medical imaging.
