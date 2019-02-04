
# train_model.py
# By: Samuel Finlayson
# 
# Trains a Resnet or InceptionV3 model for binary image classification
# For details, see: python train_model.py --help
#
# This file assumes that data for the classifier is stored per the structure in load_data.
#
# Note 1:  This uses Inception preprocessing for both Inception and Resnet.
#          This is because it is easy to deprocess, and these images are so unlike
#          ImageNet images that it didn't appear the specific preprocessing mattered to performance.
#
# Note 2: I know it is gross and unpythonic that I am importing only some modules up top
# and others inside the functions.
# Reasons: (1) I want to process args before importing keras
#          (2) Each function is only called once, so no performance tradeoff
#          (3) I want to be clear to future me which function requires which modules

import os
import sys
import argparse
import numpy as np


def load_data(batch_size, mixup, vFlip, rotation):
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.inception_resnet_v2 import preprocess_input

    train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=rotation,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            vertical_flip=vFlip)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Mixup
    if mixup:
        X_train = np.load('data/val_train_x.npy') # N x 224 x 224 x 3
        y_train = np.load('data/val_train_y.npy') 
        X_test = np.load('data/val_test_x.npy') # N x 2
        y_test = np.load('data/val_test_y.npy')
        from mixup_generator import MixupGenerator
        train_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=0.2, datagen=train_datagen)()
        validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)
    else:
        train_generator = train_datagen.flow_from_directory(
            'images/train',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)
        validation_generator = test_datagen.flow_from_directory(
            'images/val',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

    if mixup:
        n_data = (X_train.shape[0], X_test.shape[0])
    else:
        n_data = (train_generator.n, validation_generator.n)
    return (train_generator, validation_generator, n_data)

def construct_model(inceptionModel, batch_size, LR, freezeEarlyLayers = False):
    from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
    from keras.models import Model
    from keras import optimizers

    ## Data Generators 

    # Base Model
    if inceptionModel:
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top = True)
    else:
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top = True)
    outputs = base_model.layers[-2].output

    # Finetune Layer
    fine_tune_layer = Dense(128)(outputs)
    fine_tune_layer = Dropout(0.2)(fine_tune_layer) #usually .2
    fine_tune_layer = Dense(2, activation='softmax')(fine_tune_layer)

    # Final Model
    model = Model(inputs=base_model.input, outputs=fine_tune_layer)

    # Freeze early layers
    if freezeEarlyLayers:
        for layer in model.layers[:25]:
            layer.trainable = False

    # Compile Model
    model.compile(optimizer = optimizers.SGD(lr=LR, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateCallbacks(inceptionModel, nameAppend, LR, modelChecking, modelCheckpointPeriod, earlyStopping, earlyStopPatience):
    from keras import callbacks

    # Construct TB Directory
    if inceptionModel:
        runId = "InceptionV3"
    else:
        runId = "ResNet50"
    runId += "_" + nameAppend
    runId += '_LearnRate-' + str(LR)

    tb_dir0 = "./keras_logs/" + runId + "_upfront"
    if not os.path.exists(tb_dir0):
        os.makedirs(tb_dir0)
    tb_dir1 = "./keras_logs/" + runId + "_wCheckOrStop"
    if not os.path.exists(tb_dir1):
        os.makedirs(tb_dir1)

    # Tensorboard Callback
    
    callBackList = []
    tbCallBack0 = callbacks.TensorBoard(log_dir=tb_dir0, histogram_freq=0,
                                             write_graph=False, write_images=False)
    tbCallBack1 = callbacks.TensorBoard(log_dir=tb_dir1, histogram_freq=0,
                                             write_graph=False, write_images=False)
    callBackList = [tbCallBack1]

    # Model Checking Callback
    if modelChecking:
        modelCheck = callbacks.ModelCheckpoint('models/' + runId + '_weights.epoch-{epoch:02d}-val_acc-{val_acc:.4f}.hdf5',
                                                    monitor='val_acc', verbose=0,
                                                    save_best_only=True, save_weights_only=True,
                                                    mode='auto', period=modelCheckpointPeriod)
        callBackList += [modelCheck]
        
    # Early Stopping Callback
    if earlyStopping:
        earlyStop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=earlyStopPatience,
                                                  verbose=0, mode='auto')
        
        callBackList += [earlyStop]

    return ([tbCallBack0], callBackList)

def fit_model(model, callbacks, train_generator, validation_generator, n_data, batch_size, max_epochs, n_epoch_beforeSaving):

    model.fit_generator(
            train_generator,
            steps_per_epoch= n_data[0] // batch_size,
            epochs=n_epoch_beforeSaving,
            validation_data=validation_generator,
            validation_steps= n_data[1] // batch_size,
            callbacks = callbacks[0] # Begin with only tensorboard
    )

    if (max_epochs-n_epoch_beforeSaving > 0):
        model.fit_generator(
                train_generator,
                steps_per_epoch= n_data[0]  // batch_size,
                epochs= max_epochs-n_epoch_beforeSaving,
                validation_data=validation_generator,
                validation_steps= n_data[1]  // batch_size,
                callbacks = callbacks[1]
        )

    return model


if __name__ == '__main__':

    # Set up argparser
    parser = argparse.ArgumentParser(description='Train ResNet50 or InceptionResNetV2 Model')
    parser.add_argument("--incept", action='store_true', default=False, help="Train Inception Model, else train ResNet model (default: False)")
    parser.add_argument('--lr', type=float, default = 1e-3, help = "Learning rate (default: 1e-3)")
    parser.add_argument('--max_epochs', type=int, default = 300, help = "Max number epochs for which to train. (default: 300)")
    parser.add_argument("--model_checkpointing", type=int, default=10, help="Save best model checkpoints with period N.  Negative value = no checkpointing. (default: 10)")
    parser.add_argument("--early_stopping", type=int, default=100, help="Apply early stopping with patience N. Negative value = no early stopping (default: 100)")
    parser.add_argument("--min_epochs", type=int, default=200, help="Number of epochs to run before checkpoint or early stopping (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size  (default: 32)")
    parser.add_argument("--nameAppend", type=str, default="", help="Add-on to name")
    parser.add_argument("--gpu1", action='store_true', default=False, help="Use GPU 1, else use GPU zero")
    parser.add_argument("--mixup", action='store_true', default=False, help="Use mixup for data processing")
    parser.add_argument("--vFlip", action='store_true', default=False, help="Vertical Flipping during data aug")
    parser.add_argument("--rotation", type=int, default=45,  help="Degree of rotation during data aug")

    # Get Arguments
    args = parser.parse_args()
    inceptionModel = args.incept
    LR = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    modelCheckpointPeriod = args.model_checkpointing
    earlyStopPatience = args.early_stopping
    n_epoch_beforeSaving = args.min_epochs
    mixup = args.mixup
    nameAppend = args.nameAppend
    vFlip = args.vFlip
    rotation = args.rotation

    # Set CUDA Device Using Flag
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    if args.gpu1:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Handle checkpointing and early stopping
    modelChecking = (modelCheckpointPeriod >= 0)
    if modelChecking:
        print("Checkpointing models with period ", modelCheckpointPeriod)
    earlyStopping = (earlyStopPatience >= 0)
    if earlyStopping:
        print("Applying early stopping with patience ", earlyStopPatience)
    if not (modelChecking or earlyStopping):
        n_epoch_beforeSaving = 0
    else:
        print("Waiting ", n_epoch_beforeSaving, " epochs before starting checkpointing and/or saving.")

    # Get Model Type
    if inceptionModel:
        print("Using InceptionV3architecture")
        model_name = "models/InceptionV3"
    else:
        print("Using ResNet50 architecture")
        model_name = "models/ResNet50"
    if nameAppend != "":
        model_name += "_" + nameAppend 
    model_name += "_LR-" + str(LR) + "_max_epochs-" + str(max_epochs) + '_weights_final.hdf5'
    print("filename for save weights: ", model_name)
    print("\n")
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # Train the model
    train_generator, validation_generator, n_data = load_data(batch_size, mixup, vFlip, rotation)
    model = construct_model(inceptionModel, batch_size, LR)
    callbacks = generateCallbacks(inceptionModel, nameAppend, LR, modelChecking, modelCheckpointPeriod, earlyStopping, earlyStopPatience)
    model = fit_model(model, callbacks, train_generator, validation_generator, n_data, batch_size, max_epochs, n_epoch_beforeSaving)
    model.save_weights(model_name)
