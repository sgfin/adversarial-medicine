import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import sys
import os.path as osp
import numpy as np
import PIL.Image
import time

import keras
from keras import applications
from keras import backend as K
from keras.preprocessing import image



def _convert(im):
    return ((im + 1) * 127.5).astype(np.uint8)

def show(im):
    plt.axis('off')
    plt.imshow(_convert(im), interpolation="nearest")
    plt.show()
  
def load_image(image_path, size=299):
    im = PIL.Image.open(image_path)
    im = im.resize((size, size), PIL.Image.ANTIALIAS)
    if image_path.endswith('.png'):
        ch = 4
    else:
        ch = 3
    im = np.array(im.getdata()).reshape(im.size[0], im.size[1], ch)[:,:,:3]
    return im / 127.5 - 1


class Dermatology_image_loader(object):
    def __init__(self):
        self.X_test = np.load('data/test_x_preprocess_sample.npy')
        self.y_test = np.load('data/test_y_sample.npy')
        self.X_train = np.load('data/train_x_preprocess_sample.npy')
        self.y_train = np.load('data/train_y_sample.npy')
        self.true_labels = self.y_test

        
    def training_random_minibatches(self, minibatch_size):
    # number of images
        
        N = self.X_train.shape[0]
        rand_ind = np.random.permutation(N)
        X_shuffle = self.X_train[rand_ind]
        Y_shuffle = self.y_train[rand_ind]

        num_minibatches = int(N / minibatch_size)
        minibatches = []
        for n in range(num_minibatches):
            minibatch = (X_shuffle[n * minibatch_size : (n + 1) * minibatch_size], Y_shuffle[n * minibatch_size : (n + 1) * minibatch_size])
            minibatches.append(minibatch)

        # if N % minibatch_size != 0:
            # trailing_batch = (X_shuffle[num_minibatches * minibatch_size:], Y_shuffle[num_minibatches * minibatch_size:])
            # minibatches.append(trailing_batch)

        return minibatches
        
    def get_test_images(self, n_images):
        n_test = self.X_test.shape[0]
        random_indices = np.random.randint(low = 0, high = n_test, size = n_images)
        true_labels = self.y_test[random_indices]
        return self.X_test[random_indices], random_indices, true_labels

    def get_test_images_opp(self, target_label):
        """ returns test images with labels that are opposite of target_label """

        boolean_index = np.argmax(self.y_test, axis=1) != target_label
        y_test_opp = self.y_test[boolean_index]
        X_test_opp = self.X_test[boolean_index]
        # indices of True
        indices = np.where(boolean_index)[0]
        return X_test_opp, y_test_opp, indices
      
    def get_all_test_images_labels(self):
        return self.X_test, self.y_test
        


  

def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
    """
     If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
     then it maps the output point (x, y) to a transformed input point 
     (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
     where k = c0 x + c1 y + 1. 
     The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90. * (math.pi/2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
        [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale 
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image. 
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift/(2*im_scale))
    b2 = y_origin_delta - (y_shift/(2*im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

def test_random_transform(min_scale=0.5, max_scale=1.0,  max_rotation=22.5):
    """
    Scales the image between min_scale and max_scale
    """
    img_shape = [100,100,3]
    img = np.ones(img_shape)

    sess = tf.Session()
    image_in = tf.placeholder(dtype=tf.float32, shape=img_shape)
    width = img_shape[0]

    def _random_transformation():
        im_scale = np.random.uniform(low=min_scale, high=1.0)

        padding_after_scaling = (1-im_scale) * width
        x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
        y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)


        rot = np.random.uniform(-max_rotation, max_rotation)

        return _transform_vector(width, 
                                         x_shift=x_delta,
                                         y_shift=y_delta,
                                         im_scale=im_scale, 
                                         rot_in_degrees=rot)

    random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
    random_xform_vector.set_shape([8])

    output = tf.contrib.image.transform(image_in, random_xform_vector , "BILINEAR")

    xformed_img = sess.run(output, feed_dict={
        image_in: img
    })

    show(xformed_img)


#@title class ModelState()

def get_peace_mask(shape):
    path = osp.join(DATA_DIR, "peace_sign.png")
    pic = PIL.Image.open(path)
    pic = pic.resize(shape[:2], PIL.Image.ANTIALIAS)
    if path.endswith('.png'):
        ch = 4
    else:
        ch = 3
    pic = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], ch)
    pic = pic / 127.5 - 1
    pic = pic[:,:,3]

    peace_mask = (pic + 1.0) / 2
    peace_mask = np.expand_dims(peace_mask, 2)
    peace_mask = np.broadcast_to(peace_mask, shape)
    return peace_mask


def _circle_mask(shape, sharpness = 40):
    """Return a circular mask of a given shape"""
    assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

    diameter = shape[0]  
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx**2 + yy**2) ** sharpness

    mask = 1 - np.clip(z, -1, 1)
    mask = np.expand_dims(mask, axis=2)
    mask = np.broadcast_to(mask, shape).astype(np.float32)
    return mask

def gen_target_ys(batch_size, target_label=None):
    if target_label is None:
        label = TARGET_LABEL
    else:
        label = target_label

    y_one_hot = np.zeros(2)
    y_one_hot[label] = 1.0
    y_one_hot = np.tile(y_one_hot, (batch_size, 1))
    return y_one_hot





class ModelContainer():
    """Encapsulates an Imagenet model, and methods for interacting with it."""
  
    def __init__(self, model_name, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
        # Peace Mask: None, "Forward", "Backward"
        self.model_name = model_name
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.peace_mask = peace_mask
        self.patch_shape = PATCH_SHAPE
        self._peace_mask_overlay = peace_mask_overlay
        self.load_model(verbose=verbose)


    def patch(self, new_patch=None):
        """Retrieve or set the adversarial patch.

        new_patch: The new patch to set, or None to get current patch.

        Returns: Itself if it set a new patch, or the current patch."""
        if new_patch is None:
            return self._run(self._clipped_patch)

        self._run(self._assign_patch, {self._patch_placeholder: new_patch})
        return self


    def reset_patch(self):
        """Reset the adversarial patch to all zeros."""
        self.patch(np.zeros(self.patch_shape))


    def train_step(self, images=None, target_ys=None, learning_rate=5.0, scale=(0.1, 1.0), dropout=None, patch_disguise=None, disguise_alpha=None):
        """Train the model for one step.

        Args:
          images: A batch of images to train on, it loads one if not present.
          target_ys: Onehot target vector, defaults to TARGET_ONEHOT
          learning_rate: Learning rate for this train step.
          scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

        Returns: Loss on the target ys."""
        # if images is None:
        #   # images = image_loader.get_images()
        #   images, random_indices, true_labels = image_loader.get_training_images()

        if images is None:
            minibatches = image_loader.training_random_minibatches(BATCH_SIZE)

        if target_ys is None:
            target_ys = TARGET_ONEHOT

        epoch_loss = 0
        for i, minibatch in enumerate(minibatches):
            minibatch_X, minibatch_y = minibatch

            feed_dict =  {self._image_input  : minibatch_X, 
                          self._target_ys    : target_ys,
                          self._learning_rate: learning_rate}

            if patch_disguise is not None:
                if disguise_alpha is None:
                    raise ValueError("You need disguise_alpha")
                feed_dict[self.patch_disguise] = patch_disguise
                feed_dict[self.disguise_alpha] = disguise_alpha

            loss, _ = self._run([self._loss, self._train_op], feed_dict, scale=scale, dropout=dropout)
            print("(minibatch %s) loss: %s" % (i, loss))
            sys.stdout.flush()


            epoch_loss += loss / len(minibatches)

        return epoch_loss


    def inference_batch_opp(self, target_label, images=None, target_ys=None, scale=None):
        """Report loss and label probabilities, and patched images for a batch.

        Args:
          target_label: Scalar target label (either 1 or 0) with which the patch was designed
          images: A batch of images to train on, it loads if not present.
          target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present."""

        # target_y = np.argmax(target_ys, axis=1)[0]

        if images is None:
            images, true_labels, indices = image_loader.get_test_images_opp(target_label)

        n_images = images.shape[0]
        n_images = n_images // BATCH_SIZE * BATCH_SIZE

        if target_ys is None:
            # target_ys = TARGET_ONEHOT
            target_ys = gen_target_ys(target_label=target_label, batch_size = n_images)

        loss_per_example_arr, ps_arr, ims_arr = [], [], []

        for i in range(n_images // BATCH_SIZE):

            feed_dict = {self._image_input: images[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], self._target_ys: target_ys[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]}
            loss_per_example, ps, ims = self._run([self._loss_per_example, self._probabilities, self._patched_input], feed_dict, scale=scale)

            loss_per_example_arr.append(loss_per_example)
            ps_arr.append(ps)
            ims_arr.append(ims)

        loss_per_example_arr = np.concatenate(loss_per_example_arr, axis=0)
        ps_arr = np.concatenate(ps_arr, axis=0)
        ims_arr = np.concatenate(ims_arr, axis=0)
        return loss_per_example_arr, ps_arr, ims_arr, indices[:n_images]


    def load_model(self, verbose=True):

        # model = NAME_TO_MODEL[self.model_name]
        # if self.model_name in ['xception', 'inceptionv3', 'mobilenet']:
        #   keras_mode = False
        # else:
        #   keras_mode = True
        patch = None

        keras_mode = True
        self._make_model_and_ops(None, keras_mode, patch, verbose)

    def _run(self, target, feed_dict=None, scale=None, dropout=None):
        K.set_session(self.sess)
        if feed_dict is None:
            feed_dict = {}
        feed_dict[self.learning_phase] = False

        if scale is not None:
            if isinstance(scale, (tuple, list)):
                scale_min, scale_max = scale
            else:
                scale_min, scale_max = (scale, scale)
            feed_dict[self.scale_min] = scale_min
            feed_dict[self.scale_max] = scale_max

        if dropout is not None:
            feed_dict[self.dropout] = dropout
        return self.sess.run(target, feed_dict=feed_dict)


    def _make_model_and_ops(self, M, keras_mode, patch_val, verbose):
        start = time.time()
        K.set_session(self.sess)
        with self.sess.graph.as_default():
            self.learning_phase = K.learning_phase()

            # image_shape = (299, 299, 3)
            image_shape = (224, 224, 3)
            self._image_input = keras.layers.Input(shape=image_shape)

            self.scale_min = tf.placeholder_with_default(SCALE_MIN, [])
            self.scale_max = tf.placeholder_with_default(SCALE_MAX, [])
            self._scales = tf.random_uniform([BATCH_SIZE], minval=self.scale_min, maxval=self.scale_max)

            image_input = self._image_input
            self.patch_disguise = tf.placeholder_with_default(tf.zeros(self.patch_shape), shape=self.patch_shape)
            self.disguise_alpha = tf.placeholder_with_default(0.0, [])
            patch = tf.get_variable("patch", self.patch_shape, dtype=tf.float32, initializer=tf.zeros_initializer)
            self._patch_placeholder = tf.placeholder(dtype=tf.float32, shape=self.patch_shape)
            self._assign_patch = tf.assign(patch, self._patch_placeholder)

            modified_patch = patch

            def clip_to_valid_image(x):    
                return tf.clip_by_value(x, clip_value_min=-1.,clip_value_max=1.)

            if self.peace_mask == 'forward':
                mask = get_peace_mask(self.patch_shape)
                modified_patch = patch * (1 - mask) - np.ones(self.patch_shape) * mask + (1+patch) * mask * self._peace_mask_overlay

            self._clipped_patch = clip_to_valid_image(modified_patch)

            if keras_mode:
                image_input = tf.image.resize_images(image_input, (224, 224))
                image_shape = (224, 224, 3)
                modified_patch = tf.image.resize_images(patch, (224, 224))

            self.dropout = tf.placeholder_with_default(1.0, [])
            patch_with_dropout = tf.nn.dropout(modified_patch, keep_prob=self.dropout)
            patched_input = clip_to_valid_image(self._random_overlay(image_input, patch_with_dropout, image_shape))


            def to_keras(x):
                x = (x + 1) * 127.5
                R,G,B = tf.split(x, 3, 3)
                R -= 123.68
                G -= 116.779
                B -= 103.939
                x = tf.concat([B,G,R], 3)

                return x

            # Since this is a return point, we do it before the Keras color shifts
            # (but after the resize, so we can see what is really going on)
            self._patched_input = patched_input

            # if keras_mode:
                # patched_input = to_keras(patched_input)


            # Labels for our attack (e.g. always a toaster)
            # self._target_ys = tf.placeholder(tf.float32, shape=(None, 1000))
            self._target_ys = tf.placeholder(tf.float32, shape=(None, 2))


            # Load the model
            model = keras.models.load_model('models/wb_model.h5')

            if self.model_name == 'resnet2':
                model.load_weights('models/bb_weights.hdf5')

            new_input_layer = keras.layers.Input(tensor=patched_input)
            model.layers.pop(0)
            output = model(patched_input)
            model = keras.models.Model(inputs = new_input_layer, outputs = output)
            self._probabilities = model.outputs[0]
            logits = self._probabilities.op.inputs[0]      
            self.model = model   


            self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._target_ys, 
                logits=logits
            )

            # self._loss_per_example = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=self._target_ys, 
            #     logits=logits
            # )

            self._target_loss = tf.reduce_mean(self._loss_per_example)

            self._patch_loss = tf.nn.l2_loss(patch - self.patch_disguise) * self.disguise_alpha



            self._loss = self._target_loss + self._patch_loss
            # Train our attack by only training on the patch variable
            self._learning_rate = tf.placeholder(tf.float32)
            self._train_op = tf.train.GradientDescentOptimizer(self._learning_rate)\
                                     .minimize(self._loss, var_list=[patch])

            if patch_val is not None:
                self.patch(patch_val)
            else:
                self.reset_patch()


            elapsed = time.time() - start
            if verbose:
                print("Finished loading {}, took {:.0f}s".format(self.model_name, elapsed))       


    def _pad_and_tile_patch(self, patch, image_shape):
        # Calculate the exact padding
        # Image shape req'd because it is sometimes 299 sometimes 224

        # padding is the amount of space available on either side of the centered patch
        # WARNING: This has been integer-rounded and could be off by one. 
        #          See _pad_and_tile_patch for usage
        return tf.stack([patch] * BATCH_SIZE)

    def _random_overlay(self, imgs, patch, image_shape):
        """Augment images with random rotation, transformation.

        Image: BATCHx299x299x3
        Patch: 50x50x3

        """
        # Add padding

        image_mask = _circle_mask(image_shape)

        if self.peace_mask == 'backward':
            peace_mask = get_peace_mask(image_shape)
            image_mask = (image_mask * peace_mask).astype(np.float32)
        image_mask = tf.stack([image_mask] * BATCH_SIZE)
        padded_patch = tf.stack([patch] * BATCH_SIZE)

        transform_vecs = []    

        def _random_transformation(scale_min, scale_max, width):
            im_scale = np.random.uniform(low=scale_min, high=scale_max)

            padding_after_scaling = (1-im_scale) * width
            x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)


            rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)

            return _transform_vector(width, 
                                             x_shift=x_delta,
                                             y_shift=y_delta,
                                             im_scale=im_scale, 
                                             rot_in_degrees=rot)    

        for i in range(BATCH_SIZE):
            # Shift and scale the patch for each image in the batch
            random_xform_vector = tf.py_func(_random_transformation, [self.scale_min, self.scale_max, image_shape[0]], tf.float32)
            random_xform_vector.set_shape([8])

            transform_vecs.append(random_xform_vector)

        image_mask = tf.contrib.image.transform(image_mask, transform_vecs, "BILINEAR")
        padded_patch = tf.contrib.image.transform(padded_patch, transform_vecs, "BILINEAR")

        inverted_mask = (1 - image_mask)
        return imgs * inverted_mask + padded_patch * image_mask





def _convert(im):
    return ((im + 1) * 127.5).astype(np.uint8)

def show(im):
    plt.axis('off')
    plt.imshow(_convert(im), interpolation="nearest")
    plt.show()
  
def show_patch(model_or_image):
    if hasattr(model_or_image, 'patch'):
        return show_patch(model_or_image.patch())
    else:
        circle = _circle_mask((299, 299, 3))
        show(circle * model_or_image + (1-circle))



def show_patched_image(im, probs_patched_image, probs_original_image, true_label, image_index):

    text1 = 'Model prediction (patched image): ' + np.array2string(probs_patched_image, separator = ', ')
    text2 = 'Model prediction (original image): ' + np.array2string(probs_original_image, separator = ', ')
    text3 = 'True label: %d' %true_label
    text4 = 'Image index: %d' % image_index
    text = text1 + '\n' + text2 + '\n' + text3 + '\n' + text4 
    
    plt.axis('off')
    plt.imshow(_convert(im), interpolation="nearest")
    plt.text(100, -5, text,
        horizontalalignment='center',
        verticalalignment='bottom')
    plt.show()

  
def report_opp(model, target_label, target_ys=None, n_show=5, scale=0.5, show_indices=None, predict_original=False):
    """
    This function applies the patch, run prediction of patched (and unpatched) images, calculates the attack success rate, and plots the resulting patched images. The function works with images with opposite class labels.

    Args:
        model: Model to be used for prediction (ModelContainer object)
        target_label: Scalar target label (eithe 1 or 0) with which the patch was designed
        target_ys: One hot encoded target label
        n_show: Numer of images to display
        scale: Size of the patch relative to the image    
        predict_original: If True, the prediction for unpatched images will be obtained. Faster to load the result

    Returns:
        probs_patched_images: Probability prediction of model object for the patched images
        probs_original_images: Probability prediction of model object for the unpatched images
        random_indices: Indices used to suffle the test images
        true_labels: True label of the test images
        winp: Attack success rate 
    """

    # random_indices are the indices for the batch being reported
    loss_per_example, probs_patched_images, patched_imgs, indices = model.inference_batch_opp(scale = scale, target_label = target_label)

    if predict_original:
        probs_original_images, true_labels = predict_original_images()
    else:
        file_name = model.model_name + '_model_prediction_original_test_images.npy'
        probs_original_images = np.load('./etc_saved_files/' + file_name)
        probs_original_images = probs_original_images[indices]
        true_labels = np.argmax(image_loader.y_test[indices], axis=1)
        
    loss = np.mean(loss_per_example)
    # target_y = np.argmax(target_ys, axis=1)[0]
    n_images = len(indices)
    winp = (np.argmax(probs_patched_images, axis=1) == target_label).sum() / n_images

    for i in range(n_show):
        show_patched_image(patched_imgs[i], probs_patched_images[i], probs_original_images[i], true_labels[i], indices[i])

    if show_indices:
        for ind in show_indices:
            # Find the index of show_index in indices
            i = np.where(indices == ind)[0][0]
            show_patched_image(patched_imgs[i], probs_patched_images[i], probs_original_images[i], true_labels[i], indices[i])
        
    return probs_patched_images, probs_original_images, indices, true_labels, winp
  

  
def predict_original_images(indices = None):
    sess = tf.Session()
    with sess.as_default():
        model = keras.models.load_model('models/wb_model.h5')
    
        X_test, y_test = image_loader.get_all_test_images_labels()
        # probability prediction
        model_prediction_original_image = model.predict(X_test)

        # convert from onehot to 0, 1 label
        true_labels = np.argmax(y_test, axis=1)
    return model_prediction_original_image, true_labels


def train(model, target_label=1, epochs=1, learning_rate=5.0):
    """
    This function learns the patch for taget_label

    Args:
        model: Model to be trained (ModelContainer object)
        target_label: Target label for which the patch will be trained
        epochs: Number of iteration through the training set

    Returns:
        None. The trained patch can be accessed by model.patch()
    """

    model.reset_patch()

    target_ys = gen_target_ys(target_label=target_label, batch_size = BATCH_SIZE)
    
    for i in range(epochs):
        epoch_loss = model.train_step(target_ys = target_ys, scale = (0.1, 1.0), learning_rate = learning_rate)
        print("Loss after epoch %s: %s" % (i, epoch_loss))



  
def attack_opp(model, patch, target_label = 1, n_show = 5, scale = 0.4, show_indices = None, predict_original = False):
    """
    Attacks the target model with the given patch.

    Args:
        model: Target model for the attack (ModelContainer object)
        patch: Pretrained patch from a model that may be different from model (blackbox attack) (299 x 299 x 3 np array)
        target_label: Target label with which the patch was designed
        n_show: Numer of images to display
        scale: Size of the patch relative to the image
        predict_original: If True, the prediction for unpatched images will be obtained. Faster to load the result

    Returns:
        probs_patched_images: Probability prediction of model object for the patched images
        probs_original_images: Probability prediction of model object for the unpatched images
        indices: Indices of images that were used
        true_labels: True label of the test images that were used
        winp: Attack success rate 
    """

    model.reset_patch()
    model.patch(patch)

    # target_ys = gen_target_ys(target_label=target_label, batch_size = n_images)
    probs_patched_images, probs_original_images, indices, true_labels, winp = report_opp(model, target_label = target_label, n_show = n_show, scale = scale,  show_indices=show_indices, predict_original = predict_original)

    return probs_patched_images, probs_original_images, indices, true_labels, winp

  
def attack_combined(model, patch_for_0, patch_for_1, n_show=1, scale=0.4, show_indices0=None, show_indices1=None, predict_original=False):
    """
    A wrapper for attack_opp. 
    Runs attack_opp twice with target 1 and target 0, then combine the results.

    Args:
        model: Target model for the attack (ModelContainer object)
        patch_for_0: Pretrained (with target_label = 0) patch from a model that may be different from model (blackbox attack) (299 x 299 x 3 np array)
        target_label: Target label with which the patch was designed
        n_show: Numer of images to display
        scale: Size of the patch relative to the image
        show_indices0: indices of images in (entire) testset to show with target label0
        predict_original: If True, the prediction for unpatched images will be obtained. Faster to load the result

    Returns:
        probs_patched_images: Probability prediction of model object for the combined patched images
        probs_original_images: Probability prediction of model object for the combined unpatched images
        indices: Indices used to suffle the test images
        true_labels: True label of the test images
        winp: Combined attack success rate 
    """

    # Attack with target_label = 0
    probs_patched_images0, probs_original_images0, indices0, true_labels0, winp0 = attack_opp(model, patch_for_0, target_label=0, n_show=n_show, scale=scale, show_indices = show_indices0, predict_original=predict_original)

    # Attack with target_label = 1
    probs_patched_images1, probs_original_images1, indices1, true_labels1, winp1 = attack_opp(model, patch_for_1, target_label=1, n_show=n_show, scale=scale, show_indices = show_indices1, predict_original=predict_original)
    
    # Concatenate the results of two attacks (order has to be reversed)
    probs_patched_images = np.concatenate([probs_patched_images1, probs_patched_images0], axis=0)
    probs_original_images = np.concatenate([probs_original_images1, probs_original_images0], axis=0) 
    indices = np.concatenate([indices1, indices0], axis=0)
    true_labels = np.concatenate([true_labels1, true_labels0], axis=0)

    # n_images0 are images with target 1
    n_images0 = probs_patched_images0.shape[0]
    n_images1 = probs_patched_images1.shape[0]
    winp = (winp0 * n_images0 + winp1 * n_images1) / (n_images0 + n_images1)

    # correct_counts = (np.argmax(probs_patched_images0, axis=1) == 0).sum() + (np.argmax(probs_patched_images1, axis=1) == 1).sum()
    # winp_check = correct_counts / (n_images0 + n_images1)
    
    return probs_patched_images, probs_original_images, indices, true_labels, winp




  
# Global variables
image_loader = Dermatology_image_loader()
TARGET_LABEL = 1
PATCH_SHAPE = (299, 299, 3)
BATCH_SIZE = 8
TARGET_ONEHOT = gen_target_ys(BATCH_SIZE)
SCALE_MIN = 0.3
SCALE_MAX = 1.5
MAX_ROTATION = 22.5


if __name__ == "__main__":
    # Create the models
    resnet1 = ModelContainer('resnet1')
    resnet2 = ModelContainer('resnet2')

    # Loading the patch file 
    resnet1_patch_target1 = np.load('./patches/resnet1_patch_target1_epoch7.npy')
    resnet1_patch_target0 = np.load('./patches/resnet1_patch_target0_epoch1.npy')
    
    # # (Optional) Training resnet 1. Comment this out if using pretrained patch
    # train(resnet1, target_label=0, epochs=2, learning_rate=5)
    
    # Combined (attack both target labels) attack
    # Since resnet2 is used but patch was trained with resnet1, this is blackbox attack
    probs_patched_images, probs_original_images, indices, true_labels, winp = attack_combined(resnet2, patch_for_0=resnet1_patch_target0, patch_for_1=resnet1_patch_target1, n_show=1, scale=0.4, predict_original=False)

    
