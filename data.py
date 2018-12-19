import tensorflow as tf
import numpy as np
import os
import random
import cv2

def get_name_label(path):
    walks = os.walk(path)
    w_0 = next(walks)
    root = w_0[0]
    sub_dirs = w_0[1]

    cat_names = [sub_dir for sub_dir in sub_dirs]

    cat_names.sort()

    labels1 = {cat_name : idx for idx, cat_name in enumerate(cat_names)}
    labels2 = {idx : cat_name for idx, cat_name in enumerate(cat_names)}
    labels = {**labels1, **labels2}

    data_list = []

    for label, sub_dir in enumerate(sub_dirs):
        for file in os.listdir(root + '/' + sub_dir):
            _, file_extension = os.path.splitext(file)

            if file_extension == '.jpg':
                data_list.append([label, root + '/' + sub_dir + '/' + file])

    return data_list, labels

def image_reader(label, filename):
    f = tf.read_file(filename)
    image = tf.image.decode_jpeg(f, 3)

    return {'images' : image, 'labels': tf.cast(label, tf.int64)}

def read_images(data_list, batchsize, repeat = True, is_shuffle = True):

    #tf로 불러오는 코드
    random.shuffle(data_list)
    labels = [data[0] for data in data_list]
    filenames = [data[1] for data in data_list]

    d_label = tf.data.Dataset.from_tensor_slices(labels)
    d_name = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = tf.data.Dataset.zip((d_label, d_name))
    dataset = dataset.map(image_reader)

    # load on my way
    # dataset = tf.data.Dataset.from_generator(lambda: generator_multi(data_list), {
    #     'images' : tf.float32,
    #     'labels' : tf.int64
    #     },
    #     {
    #         'images' : [224, 224, 3],
    #         'labels' : []
    #     }
    # )


    if repeat:
        dataset = dataset.repeat()
    if is_shuffle:
        dataset = dataset.shuffle(batchsize * 3)
    dataset = dataset.batch(batchsize)
    return dataset



if __name__ == '__main__':
    train_data_list, label_dict = get_name_label('d:\dataset\MNIST\mnistasjpg\mini')

    train_dataset = read_images(train_data_list, 4)

    train_iter = train_dataset.make_one_shot_iterator()

    sess = tf.Session()

    while True:
        element = train_iter.get_next()
        out = sess.run(element)
        images = out['images']
        labels = out['labels']
        print(labels)

        for image in images:
            image = np.uint8(image)
            cv2.imshow('asdf', image)
            cv2.waitKey(0)