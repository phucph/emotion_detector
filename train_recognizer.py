import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from emotion_detection.config import emotion_config as config
from emotion_detection.preprocessing import ImageToArrayPreprocessor
from emotion_detection.callbacks import TrainingMonitor
from emotion_detection.callbacks import EpochCheckpoint
from emotion_detection.io import HDF5DatasetGenerator
from emotion_detection.nn import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoints directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoints to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())
# augmentation, then initialize the image preprocessor
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1 / 255.0,
                              fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap],
                                classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap],
                              classes=config.NUM_CLASSES)
# the network and compile the model
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (40, 1e-4),
    (60, 1e-5),
    # (9, 0.005),
    # (12, 0.001),
]
def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        print("lr:", lr)
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    print("lr:", lr)
    return lr
if args["model"] is None:
    print("[INFO] compiling model...")
    model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
callbacks = [EpochCheckpoint(args["checkpoints"], every=15, startAt=args["start_epoch"]),
             TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"]),
             LearningRateScheduler(lr_schedule)]
# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=75,
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1)
# close the databases
trainGen.close()
valGen.close()