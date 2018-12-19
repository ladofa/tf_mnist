import tensorflow as tf
import time
import cv2
import numpy as np
import random

import net_base
import data

import params
args = params.args



if __name__ == '__main__':
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        variables_to_init = []

        #to load load image files
        train_data_list, label_dict = data.get_name_label(args.train_image_dir)
        #eval_data_list = data.get_name_label(args.eval_image_dir)

        random.shuffle(train_data_list)
        total = len(train_data_list)
        eval_count = total // 5

        eval_data_list = train_data_list[:eval_count]
        train_data_list = train_data_list[eval_count:]

        train_dataset = data.read_images(train_data_list, args.batch_size * args.gpus)
        eval_dataset = data.read_images(eval_data_list, args.batch_size * args.gpus * 4)

        train_iter = train_dataset.make_one_shot_iterator()
        eval_iter = eval_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        next_features = iterator.get_next()

        print('build networks...')
    
        logits, loss, eqs = net_base.get_network(next_features, tf.estimator.ModeKeys.TRAIN)
        food_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mnist_test')

        #optimizer
        global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)

        variables_to_init += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_op')

        #session
        sess = tf.Session()

        #load checkpoint
        print('restore last checkpoint....')
        saver = tf.train.Saver(var_list=food_vars + [global_step])
        
        latest_filename = tf.train.latest_checkpoint('training/' + args.model)
        if latest_filename is None:
            print('init all variables...')
            variables_to_init += food_vars
            variables_to_init.append(global_step)
        else:
            print('load checkpoint : ' + latest_filename)
            saver.restore(sess, latest_filename)

        #summary
        max_image_summary = min(args.batch_size * args.gpus, 4)

        with tf.name_scope('summary'):
            sum_train_loss_ph = tf.placeholder(tf.float32, shape=[])
            sum_eval_loss_ph = tf.placeholder(tf.float32, shape=[])
            sum_eval_acc_ph = tf.placeholder(tf.float32, shape=[])
            sum_train_loss = tf.summary.scalar('train_loss', sum_train_loss_ph)
            sum_eval_loss = tf.summary.scalar('eval_loss', sum_eval_loss_ph)
            sum_eval_acc = tf.summary.scalar('eval_acc', sum_eval_acc_ph)
            summary_op = tf.summary.merge([sum_train_loss, sum_eval_loss, sum_eval_acc])
        file_writer = tf.summary.FileWriter('logs/' + args.model, sess.graph)

        #init_val    
        init_fn = tf.variables_initializer(variables_to_init)
        sess.run(init_fn)
        print('start training....')

        start_time = time.time()

        train_handle = sess.run(train_iter.string_handle())
        eval_handle = sess.run(eval_iter.string_handle())

        while True:
            _, ev_gs, train_loss = sess.run([train_op, global_step, loss], feed_dict={handle:train_handle}) 

            if ev_gs % 20 == 0:
                #초당 처리 개수
                end_time = time.time()
                ps = args.batch_size * args.gpus * 100 / (end_time - start_time)
                start_time = end_time

                #run evaluation
                eval_logits, eval_loss, eval_eqs = sess.run(
                    [logits, loss, eqs], feed_dict={handle:eval_handle})

                #TODO : calcuate accuracy
                acc = np.mean(eval_eqs)
                
                print('[gs]=%d, [t_loss]=%f, [v_loss]=%f, [ps] = %f, [acc] = %f' % (ev_gs, train_loss, eval_loss, ps, acc))

                #summary for tensorboard
                sum_dict = {}
                sum_dict[sum_train_loss_ph] = train_loss
                sum_dict[sum_eval_loss_ph] = eval_loss
                sum_dict[sum_eval_acc_ph] = acc
                summary = sess.run(summary_op, feed_dict=sum_dict)
                file_writer.add_summary(summary, ev_gs)

            if ev_gs % 1000 == 0:
                #saver.save(sess, 'training/'+ args.model + '/saved.ckpt-%d' % ev_gs)
                print('finish save : ' + str(ev_gs))
