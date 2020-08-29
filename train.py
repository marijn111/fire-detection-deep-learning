import matplotlib
matplotlib.use('Agg')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification report
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.firedetectionnet import FireDetectionNet
from pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys

def load_dataset(datasetPath):
    """
    Function for loading datasets
    """
    
    # grab the paths to all images in out dataset directory 
    imagePaths = list(paths.list_images(datasetPath))
    data = []



    for imagePath in imagePaths:
        # load and resize the image to be 128x128 pixels, ignoring aspect-ratio
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (128, 128))

        data.append(image)

    # return the data list as numpy array
    return np.array(data, dtype='float32')


ap = argparse.ArgumentParser()
ap.add_argument("-f", '--lr-find', type=int, default=0, help='whether or not to find the optimal learning rate')
args = vars(ap.parse_args())


# load the fire and non-fire data
print("[INFO] loading data...")
fireData = load_dataset(config.FIRE_PATH)
nonFireData = load_dataset(config.NON_FIRE_PATH)

# construct the class labels for the data
fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

# stack the fire data with the non-fire data, then scale the data to the range [0, 1]
data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

# perform one-hot encoding on the labels and account for skew in the labeled data
labels = to_categorical(labels, num_classes=2)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# construct the training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=config.TEST_SPLIT, random_state=42)

aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Initialize the optimzer and model
print('[INFO] compiling model...')
opt = SGD(lr=config.INIT_LR, momentum=0.9, decay-config.INIT_LR / config.NUM_EPOCHS)
model = FireDetectionNet.build(width=128, height=128, depth=3, classes=2)
model.compile(loss='binary crossentropy', optimizer=opt, metrics=['accuracy'])

# attempt to find optimal LR? initialize the learning rate finder and then train with learning
# rates ranging from 1e-10 to 1e+1
if args['lr_find'] > 0:
    print('[INFO] finding learning rate')

    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), 1e-10, 1e+1, stepsPerEpoch=ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
    epochs=20, batchSize=config.BATCH_SIZE, classWeight=classWeight)

    lrf.plot_loss()
    plt.savefig(config.LRFIND_PLOT_PATH)

    print('[INFO] learning rate finder completed')
    print('[INFO] Examine plot and adjust learning rates before training')
    sys.exit(0)

print('[INFO] training network...')
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE,
    class_weight=classWeight,
    verbose=1)


print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.CLASSES))

print(['[INFO] serializing network to "{}"...'.format(config.MODEL_PATH)]
model.save(config.MODEL_PATH))

# construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)
