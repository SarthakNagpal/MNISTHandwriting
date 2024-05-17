# MNIST Handwritten Digits Recognition

## What is MNIST?
The MNIST database (Modified National Institute of Standards and Technology database) is a large dataset of handwritten digits commonly used for training various image processing systems and machine learning models. It was created by remixing samples from NIST's original datasets to provide a standardized benchmark for digit recognition tasks.

The MNIST dataset comprises 28x28 pixel grayscale images of handwritten digits (0-9) and their corresponding labels. Each image is normalized to fit into a 28x28 pixel bounding box and anti-aliased, ensuring consistency across the dataset.

## Getting Started
To begin, we import necessary libraries and load the MNIST dataset using TensorFlow's Keras API. This dataset consists of training and testing sets containing handwritten digit images and their corresponding labels. We then preprocess the data by normalizing pixel values and reshaping the images to the required format.

## Model Architecture
We construct a convolutional neural network (CNN) model using Keras Sequential API. The model architecture consists of multiple convolutional layers followed by max-pooling layers, dropout layers for regularization, and fully connected layers. Batch normalization is applied to improve convergence speed and stability during training.

### Model Summary:
```plaintext
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 128)         73856     
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               589952    
_________________________________________________________________
batch_normalization (BatchNo (None, 128)               512       
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 730,602
Trainable params: 730,346
Non-trainable params: 256
_________________________________________________________________
```
# Training and Evaluation
We compile the model using categorical cross-entropy loss and RMSprop optimizer. During training, we apply data augmentation techniques using the ImageDataGenerator class to generate variations of the training images, enhancing model generalization. Additionally, we implement learning rate reduction using the ReduceLROnPlateau callback to adjust the learning rate dynamically based on validation performance.

The model is trained for 20 epochs using a batch size of 64. After training, the model achieves a final loss of 0.022417 and accuracy of 99.55% on the validation set.
