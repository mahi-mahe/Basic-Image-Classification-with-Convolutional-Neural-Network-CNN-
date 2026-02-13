Computer Vision Practicals – CNN & Transfer Learning

This repository documents my practical journey into image classification using TensorFlow and Keras.

Instead of only reading theory, the focus here is on building, training, and evaluating real models.
The work is divided into two parts:

Part-I: Building a Convolutional Neural Network (CNN) from scratch

Part-II: Using Transfer Learning with a pre-trained VGG16 network

Tech Stack

Python

TensorFlow 2.x

Keras

NumPy

Matplotlib

Objective

To understand how CNNs actually learn, by training one from the ground up on the CIFAR-10 dataset (10 object categories, 32×32 color images).

Dataset

The dataset is loaded directly from keras.datasets, so no manual download is required.

This practical includes:

✔ load the training and test sets
✔ normalize image pixels to the [0, 1] range
✔ convert labels into one-hot vectors
✔ define readable class names
✔ design a CNN using two convolutional blocks
✔ apply Batch Normalization and Dropout for better generalization
✔ move from feature extraction → dense layers → classification
✔ train using Adam and categorical cross-entropy
✔ evaluate performance on unseen test data
✔ visualize accuracy and loss curves
✔ run predictions on sample images
✔ keep an option to save the trained model

Model flow (simplified)

Input → Conv → BN → Conv → BN → Pool → Dropout → Conv → BN → Conv → BN → Pool → Dropout → Flatten → Dense(512) → Dropout → SoftMax

Practical 2 – Transfer Learning with VGG16 (Cats vs Dogs)

Objective

Learn how to reuse powerful features learned on ImageNet and adapt them to a new binary classification problem.

Dataset

A filtered Cats vs Dogs dataset downloaded automatically using TensorFlow utilities.

This practical includes:

✔ download and organize the dataset into train/validation folders
✔ use ImageDataGenerator to apply augmentation
✔ rescale images for validation
✔ load VGG16 without the top layers
✔ freeze the convolution base at first
✔ attach a custom classifier head
✔ train the new layers
✔ then unfreeze part of the network for fine-tuning with a smaller learning rate
✔ plot training history
✔ evaluate the final model
✔ keep an option to save the network

Model idea (simplified)

Pretrained VGG16 → GlobalAveragePooling → Dense(512) → Dropout → Sigmoid

Through these practicals, I gained hands-on experience in:

Preparing image datasets

Designing CNN architectures

Preventing overfitting

Monitoring training vs validation behavior

Using pre-trained networks effectively

Understanding when and how to fine-tune

