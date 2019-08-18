IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNEL = 3

CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
CLASSES = len(CLASS_NAMES)

VGG_MEAN = [103.94, 116.78, 123.68]

BATCH_SIZE = 64

MAX_EPOCH = 100
LOG_ITERATION = 50
VALID_ITERATION = 1000
LEARNING_RATE = 1e-4

# REPLACE_DIR = 'D:/_ImageDataset/flower_photos'
REPLACE_DIR = 'D:/_DeepLearning_DB/flower_photos'