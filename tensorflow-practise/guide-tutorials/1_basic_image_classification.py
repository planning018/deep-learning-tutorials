# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import plt_utils

print(tf.__version__)

# Import the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
"""
(60000, 28, 28)
60000
[9 0 0 ... 3 0 5]
(10000, 28, 28)
10000
"""

# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model
"""
The basic building block of a neural network is the layer. Layers extract representations from the data fed into them. 
    Hopefully, these representations are meaningful for the problem at hand.
Most of deep learning consists of chaining together simple layers. Most layers, 
    such as tf.keras.layers.Dense, have parameters that are learned during training.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
"""
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Loss function —This measures how accurate the model is during training. 
        You want to minimize this function to "steer" the model in the right direction.
Metrics —Used to monitor the training and testing steps. The following example uses accuracy, 
            the fraction of the images that are correctly classified.
"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
"""
Training the neural network model requires the following steps:
    1.Feed the training data to the model. In this example, 
        the training data is in the train_images and train_labels arrays.
    2.The model learns to associate images and labels.
    3.You ask the model to make predictions about a test set—in this example, the test_images array.
    4.Verify that the predictions match the labels from the test_labels array.
"""
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy: ', test_acc)

# make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])
"""
[2.0823032e-07 1.0930045e-10 6.0691129e-08 2.1161647e-10 5.3131144e-09
 1.7491804e-04 3.4920255e-08 3.1021836e-03 2.6279511e-07 9.9672240e-01]
9
9
"""

# Verify predictions
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt_utils.plot_image(i, predictions[i], test_labels, test_images, class_names)
plt.subplot(1, 2, 2)
plt_utils.plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt_utils.plot_image(i, predictions[i], test_labels, test_images, class_names)
plt.subplot(1, 2, 2)
plt_utils.plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt_utils.plot_image(i, predictions[i], test_labels, test_images, class_names)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plt_utils.plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Use the trained model
"""
Finally, use the trained model to make a prediction about a single image.
"""
img = test_images[1]
print(img.shape)
img = np.expand_dims(img, 0)
print(img.shape)
"""
(28, 28)
(1, 28, 28)
"""

predictions_single = probability_model.predict(img)
print(predictions_single)
plt_utils.plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))
