import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
        print('lund')

from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
### Load the data ###
my_data = pd.read_csv('C:/Users/maind/Desktop/eh/A_ZHandwrittenData.csv').astype('float32')
my_frame = pd.DataFrame(my_data)

def load_zero_nine_dataset():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    return (data, labels)

(digitsData, digitsLabels) = load_zero_nine_dataset()
print("0-9 Images Shape",digitsData.shape)
print("0-9 Labels Shape",digitsLabels.shape)

x = my_frame.drop('0', axis = 1)
y = my_frame['0']
x = np.reshape(x.values, (x.shape[0], 28, 28))
print("A-Z Images Shape",x.shape)
print("A-Z Labels Shape",y.shape)

data = np.vstack([x, digitsData])
labels = np.hstack([y, digitsLabels])
print(" Shape of Total Images ",data.shape)
print(" Shape of Total Labels ",labels.shape)


def residual_module(data, K, stride, chanDim, red=False,
                    reg=0.0001, bnEps=2e-5, bnMom=0.9):
    # the shortcut branch of the ResNet module should be
    # initialize as the input (identity) data
    shortcut = data

    # the first block of the ResNet module are the 1x1 CONVs
    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act1)

    # the second block of the ResNet module are the 3x3 CONVs
    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                   padding="same", use_bias=False,
                   kernel_regularizer=l2(reg))(act2)

    # the third block of the ResNet module is another set of 1x1
    # CONVs
    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(K, (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act3)

    # if we are to reduce the spatial size, apply a CONV layer to
    # the shortcut
    if red:
        shortcut = Conv2D(K, (1, 1), strides=stride,
                          use_bias=False, kernel_regularizer=l2(reg))(act1)

    # add together the shortcut and the final CONV
    x = add([conv3, shortcut])

    # return the addition as the output of the ResNet module
    return x

def build(width, height, depth, classes, stages, filters,reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
    # initialize the input shape to be "channels last" and the
    # channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1

    # if we are using "channels first", update the input shape
    # and channels dimension
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # set the input and apply BN
    inputs = Input(shape=inputShape)
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                           momentum=bnMom)(inputs)

    # check if we are utilizing the CIFAR dataset
    if dataset == "cifar":
        # apply a single CONV layer
        x = Conv2D(filters[0], (3, 3), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)

    # check to see if we are using the Tiny ImageNet dataset
    elif dataset == "tiny_imagenet":
        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # loop over the number of stages
    for i in range(0, len(stages)):
        # initialize the stride, then apply a residual module
        # used to reduce the spatial size of the input volume
        stride = (1, 1) if i == 0 else (2, 2)
        x = residual_module(x, filters[i + 1], stride,
                                   chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

        # loop over the number of layers in the stage
        for j in range(0, stages[i] - 1):
            # apply a ResNet module
            x = residual_module(x, filters[i + 1],
                                       (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

    # apply BN => ACT => POOL
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                           momentum=bnMom)(x)
    x = Activation("relu")(x)
    x = AveragePooling2D((8, 8))(x)

    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(reg))(x)
    x = Activation("softmax")(x)

    # create the model
    model = Model(inputs, x, name="resnet")

    # return the constructed network architecture
    return model


EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# each image in the A-Z and MNIST digts datasets are 28x28 pixels;
# however, the architecture we're using is designed for 32x32 images,
# so we need to resize them to 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# add a channel dimension to every image in the dataset and scale the
# pixel intensities of the images from [0, 255] down to [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()

labels = le.fit_transform(labels)
ounts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, stratify=None, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.05, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.15, horizontal_flip=False, fill_mode="nearest")

# initialize and compile our deep neural network
print("[INFO] compiling model...")

opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = build(32, 32, 1, len(le.classes_), (3, 3, 3),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)

# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))

# save the model to disk
print()
model.save('C:/Users/maind/Desktop/eh/OCRmodel.h5', save_format="h5")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('C:/Users/maind/Desktop/eh/plot.png')
