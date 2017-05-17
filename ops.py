import tensorflow as tf
import numpy as np
import cifar10
import os

from skimage.io import imsave


batch_size = 64
OUTPUT_PATH = 'images'



def generate_random_samples(shape):
    '''
    random generate samples with range [-1,1].
    : param shape: the shape of random generated vector
    : return: a vector
    '''
    return np.random.uniform(-1, 1, size=shape).astype(np.float32)



def random_batch(data, size):
    '''
    randomly select data with given size in dataset.
    : param data: dataset from which you select random batch.
    : param size: batch size
    : return: batch of data
    '''
    # Number of images in the training-set.
    num_images = data.shape[0]

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = data[idx, :, :, :]

    return x_batch



def batch_norm(inputs, is_train, decay=0.9):
    '''
    batch normalization. I modify code from https://github.com/fzliu/tf-dcgan/blob/master/models.py
    I came cross an annoying bugs when running original code, which indicates
    that exponential moving average cannot be under a reuse=True scope. I avoid
    this problem by removing tf.train.ExponentialMovingAverage(), and implement
    mean and variance moving by hand.
    : param input: inputs layer
    : param is_train: an indicator which indicates whether it is in training mode
    : param decay: exponential parameter
    : return: normalized batch layer
    '''

    bn_shape = inputs.get_shape()[-1]
    shift = tf.get_variable("beta", shape=bn_shape,
                            initializer=tf.constant_initializer(0.0))
    scale = tf.get_variable("gamma", shape=bn_shape,
                            initializer=tf.constant_initializer(1.0))

    pop_mean = tf.get_variable("pop_mean", shape=bn_shape,
                            initializer=tf.constant_initializer(0.0),
                            trainable=False)

    pop_var = tf.get_variable("pop_var", shape=bn_shape,
                            initializer=tf.constant_initializer(1.0),
                            trainable=False)

    def train_op():
        # training step
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    def test_op():
        # testing step
        return pop_mean, pop_var

    (mean, var) = tf.cond(is_train, train_op, test_op)

    return tf.nn.batch_normalization(inputs, mean, var, shift, scale, 1e-4)



def dense(inputs, d_outputs, name):
    '''
    fully connected layers.
    : param inputs: input layer
    : param d_outputs: dimension of output
    : name: name of scope
    : return: layer
    '''
    with tf.variable_scope(name) as scope:

        # bn = _bn(bottom, is_train) if with_bn else bottom
        w_shape = [inputs.get_shape()[-1], d_outputs]
        # inner product
        weights = tf.get_variable("weights", shape=w_shape,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        linear = tf.matmul(inputs, weights)

        # add biases
        biases = tf.get_variable("biases", shape=[d_outputs],
                                 initializer=tf.constant_initializer(0.0))
        outs = tf.nn.bias_add(linear, biases)

    return outs



def leak_relu(inputs, alpha=0.02):
    # return tf.maximum(inputs, alpha*inputs)
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)

    return f1 * inputs + f2 * abs(inputs)



def conv2d(inputs, kernel_shape, with_bn=False, is_train=None, name='conv'):
    '''
    regular convolution with batch normalization layer.
    : param inputs: inputs layer
    : kernel_shape: shape of kernel
    : with_bn: bool, implement batch normalization or not
    : is_train: is in training mode or not
    : name: name of scope
    : return: layer
    '''
    with tf.variable_scope(name) as scope:

        inputs = batch_norm(inputs, is_train) if with_bn else inputs

        weights = tf.get_variable('weights', shape=kernel_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[kernel_shape[-1]],
                                    initializer=tf.constant_initializer(0.0))

        outs = tf.nn.conv2d(inputs, weights, strides=[1,2,2,1], padding='SAME')
        outs = outs + biases

        # post batch normalization
        # if with_bn:
        #     outs = batch_norm(outs, is_train)

    return outs



def conv2d_transpose(inputs, kernel_shape, output_shape, with_bn=False, is_train=None, name='deconv'):
    '''
    regular deconvolution or convolution transpose with batch normalization layer.
    : param inputs: inputs layer
    : kernel_shape: shape of kernel
    : output_shape: shape of output layer
    : with_bn: bool, implement batch normalization or not
    : is_train: is in training mode or not
    : name: name of scope
    : return: layer
    '''

    with tf.variable_scope(name) as scope:

        inputs = batch_norm(inputs, is_train) if with_bn else inputs

        weights = tf.get_variable('weights', shape=kernel_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[kernel_shape[-2]],
                                    initializer=tf.constant_initializer(0.0))

        outs = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,2,2,1], padding='SAME')
        outs = outs + biases

        # post batch normalization
        # if with_bn:
        #     outs = batch_norm(outs, is_train)

    return outs


def generator(inputs, is_train):
    '''
    define structure of generator
    '''
    with tf.variable_scope('generator') as scope:

        rep = dense(inputs, 4*4*256, 'g_projection')
        rep = tf.reshape(rep, [-1,4,4,256])

        deconv1 = conv2d_transpose(rep, [5,5,128,256], [batch_size,8,8,128],
                                    with_bn=True, is_train=is_train, name='g_deconv1')
        deconv1 = tf.nn.relu(deconv1)


        deconv2 = conv2d_transpose(deconv1, [5,5,64,128], [batch_size,16,16,64],
                                    with_bn=True, is_train=is_train, name='g_deconv2')
        deconv2 = tf.nn.relu(deconv2)


        deconv3 = conv2d_transpose(deconv2, [5,5,3,64], [batch_size,32,32,3],
                                    with_bn=True, is_train=is_train, name='g_deconv3')
        deconv3 = tf.nn.relu(deconv3)


        outs = tf.nn.tanh(deconv3)

        return outs


def discriminator(inputs, is_train):
    '''
    define structure of discriminator
    '''

    conv1 = conv2d(inputs, [5,5,3,64], with_bn=False,
                    is_train=is_train, name='d_conv1')
    conv1 = leak_relu(conv1)


    conv2 = conv2d(conv1, [5,5,64,128], with_bn=True,
                    is_train=is_train, name='d_conv2')
    conv2 = leak_relu(conv2)


    conv3 = conv2d(conv2, [5,5,128,256], with_bn=True,
                    is_train=is_train, name='d_conv3')
    conv3 = leak_relu(conv3)


    avg_pool = tf.reduce_mean(conv3, [1, 2])

    # fully connected
    outs = dense(avg_pool, 1, name='d_projection')

    return outs



def loss_function(logits, targets):
    '''
    define loss function
    '''
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=targets)

    return tf.reduce_mean(loss)



def deprocess_and_save(batch_res, epoch, grid_shape=(8, 8), grid_pad=5):
    '''
    create an output grid to hold the images
    '''
    (img_h, img_w) = batch_res.shape[1:3]
    grid_h = img_h * grid_shape[0] + grid_pad * (grid_shape[0] - 1)
    grid_w = img_w * grid_shape[1] + grid_pad * (grid_shape[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # loop through all generator outputs
    for i, res in enumerate(batch_res):
        if i >= grid_shape[0] * grid_shape[1]:
            break

        # deprocessing (tanh)
        img = (res + 1) * 127.5
        img = img.astype(np.uint8)

        # add the image to the image grid
        row = (i // grid_shape[0]) * (img_h + grid_pad)
        col = (i % grid_shape[1]) * (img_w + grid_pad)

        img_grid[row:row+img_h, col:col+img_w, :] = img


    # save the output image
    fname = "iteration{0}.jpg".format(epoch) if epoch >= 0 else "result.jpg"
    imsave(os.path.join(OUTPUT_PATH, fname), img_grid)
