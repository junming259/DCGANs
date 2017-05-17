import cifar10
import time

from ops import *
from skimage.io import imsave
from datetime import timedelta


batch_size = 64




def read_images():
    cifar10.maybe_download_and_extract()
    images_train, cls_train, labels_train = cifar10.load_training_data()

    return images_train



def train_gan():

    # initializer dataset
    images_train = read_images()


    # create placehodler variables, number of latent variable for generator is 100
    x_data = tf.placeholder(tf.float32, [None, 32, 32, 3], name='real_data')
    x_random= tf.placeholder(tf.float32, [None, 100], name='random_init')
    is_train = tf.placeholder(tf.bool, name='is_train')


    # define generator and discriminator
    gen = generator(x_random, is_train=is_train)
    with tf.variable_scope('d_d') as scope:
        d_fake = discriminator(inputs=gen, is_train=is_train)
        scope.reuse_variables()
        d_real = discriminator(inputs=x_data, is_train=is_train)


    # loss function
    loss_g = loss_function(d_fake, tf.ones_like(d_fake))
    loss_d = loss_function(d_fake, tf.zeros_like(d_fake)) + \
                loss_function(d_real, tf.ones_like(d_real))

    # add summary
    tf.summary.scalar('loss_d', loss_d)
    tf.summary.scalar('loss_g', loss_g)

    # retrieve variables
    var_g = [item for item in tf.trainable_variables() if item.name.startswith('g')]
    var_d = [item for item in tf.trainable_variables() if item.name.startswith('d')]

    # define optimizer function
    optimizer_d = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(loss_d, var_list=var_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(loss_g, var_list=var_g)

    # define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # add to tensorboard
    writer = tf.summary.FileWriter('Tensorboard')
    writer.add_graph(session.graph)
    merge = tf.summary.merge_all()

    t1 = time.time()

    # begin training
    for i in range(20001):

        # train discriminator
        x = random_batch(images_train, batch_size)
        x_rand = generate_random_samples([batch_size, 100])

        for j in range(1):
            session.run(optimizer_d, feed_dict={x_data:x, x_random:x_rand, is_train:True})

        # train generator
        for j in range(1):
            session.run(optimizer_g, feed_dict={x_random:x_rand, is_train:True})

        # write to tensorboard
        result = session.run(merge, feed_dict={x_data:x, x_random:x_rand, is_train:False})
        writer.add_summary(result, i)

        if i%1000 == 0:
            current_loss_d = session.run(loss_d, feed_dict={x_data:x, x_random:x_rand, is_train:False})
            current_loss_g = session.run(loss_g, feed_dict={x_random:x_rand, is_train:False})
            g_images = session.run(gen, feed_dict={x_random:x_rand, is_train:False})
            print('No.{} iteration'.format(i))
            print('Current loss for discriminator: {}'.format(current_loss_d))
            print('Current loss for generator: {}'.format(current_loss_g))
            deprocess_and_save(g_images, i)
            t2 = time.time()
            time_dif = t2 - t1
            print('Time usage: {}...'.format(timedelta(seconds=int(time_dif))))
            print()




# main function for training
train_gan()
