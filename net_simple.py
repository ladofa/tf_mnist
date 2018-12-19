import tensorflow as tf


def get_network(features, mode):
    images = features['images']
    labels = features['labels']
    if mode == tf.estimator.ModeKeys.TRAIN:
        training_flag = True
    else:
        training_flag = False

    images = tf.cast(images, tf.float32) / tf.constant([128.0]) - 1

    x = tf.reshape(images, [-1, 28 * 28 * 3])
    x = tf.layers.dense(x, 100, tf.nn.relu)
    x = tf.layers.dense(x, 10) 
    logits = x
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    return logits, loss