import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display some sample images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

"""
Input imaged have height and width of 32 pixels and 3 colour channels (RGB)

Convolutional layer apply learnable kernels ot the input image, which slide over the image 
performing dot products on the input and the filter; this process extracts local features, 
e.g. edges, textures and simple patterns. Parameterised by:

- There are 32 filters in the layer, of size 3x3 pixels, each filter will learn about different 
    features in the data, e.g. one may learn to detect vertical edges, another corners
- The Rectified Linear Unit (ReLU) activation function introduces non-linearity to help the
    network learn more complex patterns
- This layer will produce an output tensor (stack of feature maps), highlighting different 
features of the input image
"""

"""
Reduce spacial dimension of the feature maps, while retaining most important info, 
to reduce computational complexity and overfitting. Parameterised by:

- `(2, 2)` Size of pooling window; this will output the maximimum value for each 2x2 window 
to downsample the feature map by a factor of two
"""

"""The above convolutional and pooling steps are repeated, to increase the abstraction of the 
image features
"""

"""
Converts 2D feature maps into 1D vector, before passing into the Dense layer
"""

"""
These 'fully connected layers) perform high-level reasoning and make final predictions based 
on the features extracted by the convolutional and pooling layers. Parameterised by:
- `64` neurons in the first dense layer.

Processes the flattened vector, applying ReLU
"""

"""
10-D vector where each element is a predicted score for each class
`10` neurons in the final dense layer, corresponding to 10 classes in the input dataset
"""

# Build the convolutional neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Display the model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

"""
Reducing overfitting:

1. Data augmentation to increase diversity of the training data
2. Dropout of random neurons to reduce overreliance
3. Regularisation: penalise large weights, thereby encouraging simpler models
4. Early stopping of training
5. Simplifying
6. Larger dataset
"""