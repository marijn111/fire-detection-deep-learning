import os

# define paths to fire and non fire datasets
FIRE_PATH = os.path.sep.join(['Robbery_Accident_Fire_Database2', 'Fire'])
NON_FIRE_PATH = 'spatial_envelope_256x256_static_8outdoorcategories'

# Class labels in the dataset
CLASSES = ['Non-Fire', 'Fire']

# define size of training and testing split
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

# define LR, batch-size, num Epochs
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

# set path to serialized model after training
 MODEL_PATH = os.path.sep.join(['output', 'fire_detection.model'])

 # define path to LR finder plot and training history plot
 LRFIND_PLOT_PATH = os.path.sep.join(['output', 'lrfind_plot.png'])
 TRAINING_PLOT_PATH = os.path.sep.join(['output', 'training_plot.png'])

 # define output path to store labels and annotations and num of images to sample
 OUTPUT_IMAGE_PATH = os.path.sep.join(['output', 'examples'])
 SAMPLE_SIZE = 50