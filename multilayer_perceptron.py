#
# Multilayer perceptron
# ......................................................................................................................

# ======================================================================================================================
# Imports

# Import all required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import mnist data set
mnist = tf.keras.datasets.mnist

# Import helper functions
from visual_helper import set_floatingpoint_formatter, draw_test_images_and_prediction_charts


# ======================================================================================================================
# Functions


# ----------------------------------------------------------------------------------------------------------------------
# Prepare the provided data set
# ----------------------------------------------------------------------------------------------------------------------
def prepare_data_set(mnistDataset):

    # Load the data set and store it into numpy arrays train_images,train_labels resp. test_images, test_labels
    (train_images, train_labels), (test_images, test_labels) = mnistDataset.load_data()

    # Input standardization
    #   Convert the samples from integers (Color values from 0-255) to floating-point numbers (Color values 0-1)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Example to see the converted color values: print(test_images[0])
    return (train_images, train_labels), (test_images, test_labels)


# ----------------------------------------------------------------------------------------------------------------------
# Now define a neuronal network model
# ----------------------------------------------------------------------------------------------------------------------
def build_multilayer_perceptron():
    # Define layer structure:
    #   2 hidden layers with each 100 neurons, sigmoid activation functions in each layer and Dropout regularization
    #   the layers are densely or fully connected
    mlp = keras.models.Sequential([
        # unstack array
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # Hidden layer 1, e.g 500 nodes
        tf.keras.layers.Dense(500, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),

        # Hidden layer 2, e.g. 250 nodes
        tf.keras.layers.Dense(250, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),

        # Output layer
        tf.keras.layers.Dense(10, activation='sigmoid'),
    ])
    # Define learning rule:
    #   adam: advanced form of gradient descent
    #   binary crossentropy: logit cost function
    mlp.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return mlp

# ======================================================================================================================
# Main program


# ----------------------------------------------------------------------------------------------------------------------
# Train and evaluate multilayer perceptron with prepared data
# ----------------------------------------------------------------------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = prepare_data_set(mnist)

# Define the expected class names. For example, the class numbered 3 is labelled '3' (could be a 'Number 3' or similar)
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# The class labels are coded with integer values from 0 to 9 and need to be converted to one-hot encoding
#   one-hot: 1d array where the index numbers indicate the class numbers and binary values determine the target class
#   class 5 =[0 0 0 0 0 1 0 0 0 0]
train_labels_onehot = keras.utils.to_categorical(train_labels)
test_labels_onehot = keras.utils.to_categorical(test_labels)

mlp = build_multilayer_perceptron()

# Train the model using the imported images and labels and parameters
mlp.fit(train_images, train_labels_onehot, epochs=10, shuffle=True)

# Evaluate trained model with test subset
test_loss, test_acc = mlp.evaluate(test_images, test_labels_onehot)


# Print accuracy of trained model evaluated with test set
print('\nTest accuracy:', test_acc * 100, "%")

# ----------------------------------------------------------------------------------------------------------------------
# Illustration of calculated results with an example
# ----------------------------------------------------------------------------------------------------------------------

# Print the prediction of first test image
set_floatingpoint_formatter()
predictions = mlp.predict(test_images)
print('\nArray indexed according to the class labels with confidence percentages for first test image: ', predictions[0])
print('Highest confidence value: ', np.argmax(predictions[0]))
print('True label: ', test_labels[0])


# ----------------------------------------------------------------------------------------------------------------------
# Visual illustration of results with the first [image_count] digits
# ----------------------------------------------------------------------------------------------------------------------
ROW_COUNT = 5
COL_COUNT = 5
IMAGE_COUNT = ROW_COUNT * COL_COUNT
draw_test_images_and_prediction_charts(test_labels, class_labels, predictions, test_images, ROW_COUNT, COL_COUNT)

