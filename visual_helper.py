
# ======================================================================================================================
# Imports

import numpy as np
from matplotlib import pyplot as plt

# ======================================================================================================================
# Constants

COLOR_RED='#e01111'
COLOR_ORANGE='orange'
COLOR_GREEN='green'

# ======================================================================================================================
# Helper Functions

# ----------------------------------------------------------------------------------------------------------------------
# Reducing decimal places of numpy array values when printing to two
# ----------------------------------------------------------------------------------------------------------------------
def set_floatingpoint_formatter():
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})

# ----------------------------------------------------------------------------------------------------------------------
# Image and description
# ----------------------------------------------------------------------------------------------------------------------
def draw_image(prediction, test_label, test_image, class_label):
    # x and y-axis has no description since it is an image
    plt.xticks([])
    plt.yticks([])
    # test_image contains the pixel color values and they are displayed on a scale between black and white
    plt.imshow(test_image, cmap=plt.cm.binary)
    # If the prediction(label with the highest confidence value) matches the true label, the text color is set to green
    # and otherwise to red
    predicted_label = np.argmax(prediction)
    if predicted_label == test_label:
        color = COLOR_GREEN
    else:
        color = COLOR_RED
    # print the confidence percentage and comparison of predicted label vs true label
    plt.xlabel("Predicted label {} with {:2.0f}% confidence \nTrue label: {}"
               .format(class_label[predicted_label], 100 * np.max(prediction), class_label[test_label]),
               fontdict={'color':color, 'fontsize':7})

# ----------------------------------------------------------------------------------------------------------------------
# probability chart
# ----------------------------------------------------------------------------------------------------------------------
def draw_probability_chart(prediction, test_label):
    plt.grid(True)
    plt.tick_params(size=6, color='gray')
    # lettering of x and y axis
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=6,color='gray')
    plt.yticks([0, 0.5, 1.0], size=6, color='gray')
    # draw the bars for each confidence level in orange and set the graphs ceiling to y=1
    chart = plt.bar(range(10), prediction, color=COLOR_ORANGE)
    plt.ylim([0, 1])
    # paint the prediction red as default
    predicted_label = np.argmax(prediction)
    chart[predicted_label].set_color(COLOR_RED)
    # repaint if it matches the true_label
    chart[test_label].set_color(COLOR_GREEN)

# ----------------------------------------------------------------------------------------------------------------------
# draw the images with their chart
# ----------------------------------------------------------------------------------------------------------------------
def draw_test_images_and_prediction_charts(test_labels, class_labels, predictions, test_images,
                                           img_and_bar_charts_rows, img_and_bar_charts_columns):
    # each test run has an image and a chart --> columns*2
    grid_col_count = img_and_bar_charts_columns * 2
    grid_row_count = img_and_bar_charts_rows
    num_images = img_and_bar_charts_rows * img_and_bar_charts_columns

    # set size of final figure in inches
    plt.figure(figsize=(32, 14))
    for i in range(num_images):
        # plot the grid with the dimensions GRID_ROW_COUNT*GRID_COL_COUNT and go along the indexes
        plt.subplot(grid_row_count, grid_col_count, 2 * i + 1)
        # draw the image at the grid location of the current index
        draw_image(predictions[i], test_labels[i], test_images[i], class_labels)
        # go to the next index and draw the chart belonging to the image
        plt.subplot(grid_row_count, grid_col_count, 2 * i + 2)
        draw_probability_chart(predictions[i], test_labels[i])
    plt.show()



