
# craft_attakcs.py
# By: Samuel Finlayson
# 
# Crafts and resports results of black and white box PGD attacks
# For details of arguments, run: python craft_attakcs.py --help
#
# This file assumes that data for the classifier is stored per the structure in load_data.
#
# Note 1:  This currently hardcodes the location of the data, but is easy to tweak to change this
#
# Note 2:  This assumes that images were preprocessed using the inception preprocessing function
#
# Note 3: I know it is gross and unpythonic that I am importing only some modules up top
# and others inside the functions.
# Reasons: (1) I want to process args before importing keras
#          (2) Few functions are called more than once, so little performance tradeoff
#          (3) I want to be clear to future me which function requires which modules

import argparse
import os

import numpy as np
import scipy.stats as st
import scipy.misc
import time
import sys

from sklearn import metrics
from sklearn.metrics import auc
from copy import copy

# Undo the inception preprocessing
def deprocess_inception(y, rescale = False):
    x = copy(y).astype(np.float)
    x += 1.
    x /= 2.
    if rescale:
        x *= 255.
    return x

# For computing summary stats: returns mean and 95% CI
def mean_ci(x):
    mn = np.mean(x)
    ci = st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x))
    return (mn, ci[0], ci[1])

# Load the data and optionally apply inception preprocessing
def load_data(limit100, preprocess = False, x_path = 'data/val_test_x_preprocess.npy', y_path = 'data/val_test_y.npy'):
    print("Loading Data ....")
    X_test = np.load(x_path, mmap_mode = "r")
    y_test = np.load(y_path)

    print("Loaded.")
    if limit100:
        print("Shrinking data to 50 samples per class")
        X_test = np.concatenate((X_test[0:50], X_test[-50:]))
        y_test = np.concatenate((y_test[0:50], y_test[-50:]))
    
    if preprocess:
        from keras.applications.inception_resnet_v2 import preprocess_input
        X_test = preprocess_input(X_test)

    return(X_test, y_test)

# Load model and optionally model weights from file 
def build_model(pathModel, pathWeights = None):
    from keras.models import load_model
    from cleverhans.utils_tf import initialize_uninitialized_global_variables

    print("Loading model from disk...")
    model = load_model(pathModel)
    if pathWeights:
        print("Loading model weights")
        model.load_weights(pathWeights)

    print("Loaded.")
    K.set_learning_phase(0)
    initialize_uninitialized_global_variables(sess)

    #from keras import optimizers
    # model.compile(optimizer = optimizers.SGD(lr=1e-3, momentum=0.9),
    #           loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Craft PGD attack against the model in cleverhans
def createAttack(model, sess, x, y, X_test, y_test, eps = 0.02):
    from cleverhans.attacks import MadryEtAl

    print("Beginning PGD attack")
    pgd = MadryEtAl(model, back='tf', sess=sess)
    preds = model(x)

    t0 = time.time()
    batch_size = 64

    # Incredibly horrible and ugly way to iterate over X_test.  Sorry.
    X_test_adv_pgd = np.zeros(X_test.shape)
    num_batches = X_test.shape[0] // batch_size
    for i in range(X_test.shape[0] // batch_size):
        batch_start = batch_size*i 
        batch_end = batch_size*(i+1)
        batch = X_test[batch_start:batch_end]
        if not (i % 20):
            print("attacking batch", i, "from ", batch_start, " to ", batch_end, file=sys.stderr)
        attack_target = 1 - y_test[batch_start:batch_end]
        pgd_params = {'eps': eps,
                  'eps_iter': 0.01,
                  'clip_min': -1.,
                  'clip_max': 1.,
                  'nb_iter': 20,
                  'y_target': attack_target}
        X_test_adv_pgd[batch_start:batch_end] = pgd.generate_np(batch, **pgd_params)
    if X_test.shape[0] % batch_size:
        batch_start = (num_batches * batch_size )
        batch_end = X_test.shape[0]
        batch = X_test[batch_start:batch_end].reshape((-1,224,224,3))
        print("attacking residual batch from ", batch_start, " to ", batch_end, file=sys.stderr)
        attack_target = 1 - y_test[batch_start:batch_end].reshape((-1,2))
        pgd_params = {'eps': eps,
                  'eps_iter': 0.01,
                  'clip_min': -1.,
                  'clip_max': 1.,
                  'nb_iter': 20,
                  'y_target': attack_target}
        X_test_adv_pgd[batch_start:batch_end] = pgd.generate_np(batch, **pgd_params)

    # Report on timing
    t1 = time.time()
    total = t1-t0
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    print ("Completed attack in %d:%02d:%02d" % (h, m, s))
    
    return X_test_adv_pgd

# Print summary of results given predictions and labels
def printResults(model_preds, y_test):
    acc = np.mean(np.round(model_preds)[:,0] == y_test[:,0])
    print('Test accuracy: %0.4f' % acc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], model_preds[:,1])
    auc_score = auc(fpr,tpr)
    print('AUC: %0.4f' % auc_score)

    conf = mean_ci(np.max(model_preds, axis = 1))
    print('Avg. Confidence: ' + '{0:.6f} '.format(conf[0]) + \
          '({0:.6f}'.format(conf[1]) + ' - {0:.6f})'.format(conf[2]))

# Evaluate the specified label on the clean and adversarial data and preint results
def evaluate(model, sess, x, X_test, y_test, X_test_adv, attackType, print_image_index = [], testClean = False, saveFlag = True):
    from cleverhans.utils_tf import batch_eval

    eval_par = {'batch_size': 32}

    # Optionally test on clean examples
    if testClean:
        print("Clean Examples:")
        model_preds_clean = batch_eval(sess, [x], [model(x)], [X_test], args=eval_par)[0]
        printResults(model_preds_clean, y_test)
        print("")
    else:
        print("(Skipping clean examples)\n")

    # Evaluate results on adversarial examples
    print("Adversarial Examples:")
    model_preds = batch_eval(sess, [x], [model(x)], [X_test_adv], args=eval_par)[0]
    printResults(model_preds, y_test)

    if saveFlag:
        np.save( "data/pgd_preds_" + attackType + ".npy" , model_preds)

    # Calculate L2 norm of pertrubations
    l2_norm = np.sum((X_test_adv - X_test)**2, axis=(1, 2, 3))**.5
    l2_norm_sum = mean_ci(l2_norm)
    print('Avg. L2 norm of perturbations: ' + '{0:.6f} '.format(l2_norm_sum[0]) + \
     '({0:.6f}'.format(l2_norm_sum[1]) + ' - {0:.6f})'.format(l2_norm_sum[2]))

    # Identify the most perturbed images from health and sick patients
    indMaxDiff_healthy = np.argmax(l2_norm[y_test[:,1] == 0])
    indMaxDiff_sick = np.argmax(l2_norm[y_test[:,1] == 1])
    indMaxDiff_sick_shifted = indMaxDiff_sick + np.nonzero(y_test[:,1] == 1)[0][0]
    print("Most perturbed images are " + str(indMaxDiff_healthy) + " and " + str(indMaxDiff_sick_shifted))

    # Optionally save the most perturbed images and also any images whose indices are in print_image_index
    if saveFlag:
        for ind in print_image_index:
            scipy.misc.imsave('example_images/normal_img_' + str(ind) + '.png',
                              deprocess_inception(X_test[ind]))
            scipy.misc.imsave('example_images/attack_pgd_img' + str(ind) + attackType + '.png',
                              deprocess_inception(X_test_adv[ind]))

        scipy.misc.imsave('example_images/biggest_attack_' + attackType + '_img' + str(indMaxDiff_healthy) + '.png',
                          deprocess_inception(X_test_adv[indMaxDiff_healthy]))
        scipy.misc.imsave('example_images/biggest_attack_' + attackType + '_img' + str(indMaxDiff_sick_shifted) + '.png',
                          deprocess_inception(X_test_adv[indMaxDiff_sick_shifted]))

    return


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(description='Build adversarial attakcs')
    parser.add_argument("--pathModel", type=str, default="models/wb_model.h5", 
        help="Path to .h5 file for true model (default: models/wb_model.h5)")
    parser.add_argument("--pathWBWeights", type=str, default=None, #models/wb_weights.hdf5
        help="Optionally provide path to weights to load into WB model (default: None)")
    parser.add_argument("--pathBBWeights", type=str, default="models/bb_weights.hdf5", 
        help="Path to weights for independent BB model (default: models/bb_weights.hdf5)")
    parser.add_argument("--pathDataX", type=str, default="data/val_test_x_preprocess.npy", 
        help="Path to training data X (default: data/val_test_x_preprocess.npy)")
    parser.add_argument("--pathDataY", type=str, default="data/val_test_y.npy", 
        help="Path to training labels y (default: data/val_test_y.npy)")
    parser.add_argument("--limit100", action='store_true', default=False, help="Only attack 50 examples from each class")
    parser.add_argument('--eps', type=float, default = 0.02, help = "Epsilon parameter of PGD (default: 0.02)")
    parser.add_argument("--filename", type=str, default="data/pgd_eps02_", help="Base of filename to save model (default: 'data/pgd_eps02_')")
    parser.add_argument("--print_image_index", type=int, nargs = "*", default=None, help="Image indices to save to file in addition to the most perturbed. (default: None)")
    parser.add_argument("--gpu1", action='store_true', default=False, help="Use GPU 1, else use GPU zero")
    parser.add_argument("--dontSaveResults", action='store_true', default=False, help="Save results (default: true)")

    # Get Arguments
    args = parser.parse_args()
    eps = args.eps
    filename = args.filename
    pathModel = args.pathModel
    pathBBWeights = args.pathBBWeights
    pathWBWeights = args.pathWBWeights
    pathDataX = args.pathDataX
    pathDataY = args.pathDataY
    limit100 = args.limit100
    saveResults = not args.dontSaveResults
    print_image_index = args.print_image_index
    if limit100:
        filename += "small_"

    # Set Up GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    if args.gpu1:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    # Import tensorflow/keras
    import tensorflow as tf
    import keras
    from keras import backend as K

    # Create Session
    sess = tf.InteractiveSession()
    K.set_session(sess)
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    # Run attack
    (X_test, y_test) = load_data(limit100, x_path = pathDataX, y_path = pathDataY)

    # Black box
    print("\nBlack Box Attack:")
    model = build_model(pathModel, pathBBWeights)
    X_test_adv_bb = createAttack(model, sess, x, y, X_test, y_test, eps)
    if saveResults:
        print("Saving to : ", filename + 'BlackBox.npy')
        np.save(filename + 'BlackBox.npy', X_test_adv_bb)

    # White box
    print("\n\nWhite Box Attack:")
    model = build_model(pathModel, pathWBWeights)
    X_test_adv_wb = createAttack(model, sess, x ,y, X_test, y_test, eps)
    if saveResults:
        print("Saving to : ", filename + 'WhiteBox.npy')
        np.save(filename + 'WhiteBox.npy', X_test_adv_wb)

    # Evaluate
    print("\n_____Evaluate White Box Attack_____")
    evaluate(model, sess, x, X_test, y_test, X_test_adv_wb, "pgdWhiteBox", testClean = True, saveFlag = saveResults) #print_image_index

    print("\n_____Evaluate Black Box Attack_____")
    evaluate(model, sess, x, X_test, y_test, X_test_adv_bb, "pgdBlackBox", saveFlag = saveResults) #print_image_index
