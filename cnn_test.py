import tensorflow as tf
import model_build.cnn_build as cnn
from utils.dataset_generator import ImageDataset, get_record_names
from datetime import timedelta
from sys import stdout
from time import time
from math import ceil

NUM_CLASSES = 9

test_batch_size = 128
test_disp_freq = 400

# PATH ARGUMENTS
tf.app.flags.DEFINE_string('data_records_dir', None, 'Directory of TFRecord files')

tf.app.flags.DEFINE_string('restore_path', None, 'Path to model.ckpt file for weight initialization')

# GPU ID
tf.app.flags.DEFINE_integer('gpu_id', '0','ID of GPU to use')

ARGS = tf.app.flags.FLAGS

gpu_id = ARGS.gpu_id

data_records_dir = ARGS.data_records_dir
restore_path = ARGS.restore_path

def cnn_test():

    test_records = get_record_names(data_records_dir, split_name='test')
    num_test_frames = 636529
    test_batches_per_epoch = ceil(num_test_frames / float(test_batch_size))

    with tf.device('/cpu:0'):
        test_images = ImageDataset(test_records, batch_size=test_batch_size,
                                          split_name='test').get_dataset()

        test_iterator = test_images.make_one_shot_iterator()
        image_batch, label_batch, _, _, _ = test_iterator.get_next()

    with tf.device('/device:GPU:{0}'.format(gpu_id)):

        is_train = tf.placeholder(shape=(), dtype=tf.bool, name='is_test')
        model = cnn.VGGNet(inputs=image_batch, true_labels=label_batch, is_train=is_train, num_classes=NUM_CLASSES)
        accuracy = model.test()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    with tf.Session(config=config) as session:

        session.run(init_op)
        if restore_path != None:
            saver.restore(session, restore_path)

        test_feed_dict = {is_train: False}

        #============================================== TESTING ==================================================#

        batch_counter = 0
        start_time = time()
        avg_acc = []
        avg_accuracy = 0
        while True:

            stdout.write('\r>>Testing batch {0:3d}/{1:3d}....'.format(batch_counter + 1, test_batches_per_epoch))
            stdout.flush()
            acc = session.run(accuracy, feed_dict=test_feed_dict)
            avg_accuracy += acc
            avg_acc.append(acc)
            batch_counter += 1

            if (batch_counter % test_disp_freq == 0) or (batch_counter == test_batches_per_epoch):
                time_diff = timedelta(seconds=int(round(time() - start_time)))
                stdout.write('\rTEST   |   ACCURACY: {0:6.2f}   |   Time taken: {1}\n'.
                             format(100 * (sum(avg_acc) / len(avg_acc)), time_diff))

                avg_acc = []
                start_time = time()

            if batch_counter == test_batches_per_epoch:
                break

        stdout.write('\rTEST   |   OVERALL ACCURACY: {0:6.2f}'.format(100 * avg_accuracy / num_test_frames))


if __name__ == '__main__':
    cnn_test()

