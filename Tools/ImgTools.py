import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def bgr2rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


# img to matrix
def image2matrix(img_path):
    img = Image.open(img_path)
    matrix = np.asarray(img)
    return matrix.astype(np.float32)/255


# matrix to img
def matrix2image(matrix):
    matrix = matrix * 255
    img = Image.fromarray(matrix.astype(np.uint8))
    return img


# show img
def show_img(img, img_type='Greys'):
    plt.imshow(img, img_type)
    plt.axis('off')
    plt.show()


def plot_img(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# show training history
def show_history(history):
    plt.figure()
    # plt.plot(np.arange(0, 20), history["loss"], label="train_loss")
    # plt.plot(np.arange(0, 20), history["val_loss"], label="valid_loss")
    plt.plot(np.arange(0, 20), history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, 20), history["val_accuracy"], label="valid_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("# Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
