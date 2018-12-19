import tensorflow as tf

import params
args = params.args

import net_simple

def network_selector(features, mode):
    if args.model == 'simple':
        logit, loss = net_simple.get_network(features, mode)
    # else args.model = 'mb2':
    #     logit, loss = net_mb2.get_network(features, mode)
    
    return logit, loss

def get_network(features, mode):
    """ common struct for all kind """
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        gpus = 1
        with tf.variable_scope('food', reuse=tf.AUTO_REUSE):
            logit, loss = network_selector(features, mode)
    else:
        # split features for MULTI GPU
        logits = []
        losses = []

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.variable_scope('mnist_test'):
                logit, loss = network_selector(features, mode)
                logits += [logit]
                losses += [loss]

        logit = tf.concat(logits, axis=0)        
        loss = tf.reduce_mean(losses)
        loss = tf.identity(loss, 'loss')

    logit = tf.identity(logit, 'logits')

    out_label = tf.argmax(logit, axis=1)
    eqs = tf.equal(out_label, features['labels'], name='eqs')

    return logit, loss, eqs