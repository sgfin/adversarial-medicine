from craft_attack_patch import *


def mean_ci(x):
    import scipy.stats as st
    mn = np.mean(x)
    ci = st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x))
    return (mn, ci[0], ci[1])

def printResults(model_preds, y_test):
    import scipy.stats as st
    acc = np.mean(np.round(model_preds)[:,0] == y_test[:,0])
    print('Test accuracy: %0.4f' % acc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], model_preds[:,1])
    auc_score = auc(fpr,tpr)
    print('AUC: %0.4f' % auc_score)

    conf = mean_ci(np.max(model_preds, axis = 1))
    print('Avg. Confidence: ' + '{0:.6f} '.format(conf[0]) + \
          '({0:.6f}'.format(conf[1]) + ' - {0:.6f})'.format(conf[2]))


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.chdir("/home/sgf2/DBMI_server/adversarial_attacks/retinopathy/")

    # Hyperparameters
    epochs = 7
    learning_rate = 5.0

    # Load the models
    resnet1 = ModelContainer('resnet1')
    resnet2 = ModelContainer('resnet2')

    # sess = tf.Session()
    # with sess.as_default():
    #     model = keras.models.load_model('models/wb_model.h5')
    #     X_test, y_test = image_loader.get_all_test_images_labels()

    #     # probability prediction
    #     model_prediction_original_image = model.predict(X_test)

    #     # convert from onehot to 0, 1 label
    #     true_labels = np.argmax(y_test, axis=1)

    # np.save('./etc_saved_files_patch/resnet1_model_prediction_original_test_images.npy', model_prediction_original_image)

    # White Box
    model = resnet1

    # Target 0
    train(model, target_label=0, epochs=epochs, learning_rate=learning_rate)
    file_name = './patches/resnet1_patch_target0_epoch' + str(epochs) + '_wb.npy'
    np.save(file_name, model.patch())

    # Target 1
    train(model, target_label=1, epochs=epochs, learning_rate=learning_rate)
    file_name = './patches/resnet1_patch_target1_epoch' + str(epochs) + '_wb.npy'
    np.save(file_name, model.patch())

    # Black Box
    model = resnet2

    # Target 0
    train(model, target_label=0, epochs=epochs, learning_rate=learning_rate)
    file_name = './patches/resnet1_patch_target0_epoch' + str(epochs) + '_bb.npy'
    np.save(file_name, model.patch())

    # Target 1
    train(model, target_label=1, epochs=epochs, learning_rate=learning_rate)
    file_name = './patches/resnet1_patch_target1_epoch' + str(epochs) + '_bb.npy'
    np.save(file_name, model.patch())


