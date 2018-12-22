import matplotlib
import matplotlib.pyplot as plt
from utils.train_utils import *
from utils.test_utils import plot_confusion_matrix
from datetime import timedelta
from utils.dataset_generator import VideoFeatureDataset, get_record_names
from sys import stdout
from time import time
from model_build.lstm_build import LSTM
from math import ceil
from sklearn.metrics import confusion_matrix
import numpy as np

NUM_CLASSES = 9

test_batch_size = 1
max_timestep = 20000
pred_type = 'last'

# PATH ARGUMENTS
tf.app.flags.DEFINE_string('data_records_dir', None, 'Directory of TFRecord files')
tf.app.flags.DEFINE_string('output_dir', None, 'Path to save results')
tf.app.flags.DEFINE_string('restore_path', None, 'Path to model.ckpt file for weight initialization')


# GPU ID
tf.app.flags.DEFINE_integer('gpu_id', '0','ID of GPU to use')

ARGS = tf.app.flags.FLAGS

gpu_id = ARGS.gpu_id

data_records_dir = ARGS.data_records_dir
restore_path = ARGS.restore_path
output_dir = ARGS.output_dir

tokens = restore_path.split('/')
save_file = os.path.join(output_dir, '15min_{0}_{1}_{2}_{3}.npy'.format(pred_type, tokens[-4], tokens[-3], tokens[-2]))

def lstm_test():

    test_records = get_record_names(data_records_dir, split_name='test')
    test_batches_per_epoch = ceil(len(test_records) / float(test_batch_size))

    lstm_dim = int(restore_path.split('/')[-2].split(',')[0].split('=')[1])
    with tf.device('/cpu:0'):
        test_data = VideoFeatureDataset(filenames=test_records,
                                        batch_size=test_batch_size,
                                        max_timestep=max_timestep,
                                        type='TFRecord',
                                        split_name='test',
                                        feat_type='vgg_16').get_dataset()

        test_iterator = test_data.make_one_shot_iterator()
        video_batch, label_batch, _, _, _ = test_iterator.get_next()

    with tf.device('/device:GPU:{0}'.format(gpu_id)):

        model = LSTM(input=video_batch, true_labels=label_batch, num_hidden=lstm_dim,
                     num_classes=NUM_CLASSES, name='LSTM')
        accuracy, y_pred_cls = model.test(pred_type=pred_type)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver(tf.trainable_variables())
    acc_vs_frames_dict = {}

    true_positives = np.zeros([NUM_CLASSES, 900])
    cls_wise_gt_count = np.zeros([NUM_CLASSES])

    for i in range(900):
        acc_vs_frames_dict[i] = []

    with tf.Session(config=config) as session:

        saver.restore(session, restore_path)
        batch_counter = 0
        while True:
            stdout.write('\r>>Testing batch {0:3d}/{1:3d}....'.format(batch_counter + 1,
                                                                      test_batches_per_epoch))
            stdout.flush()
            acc, gt_cls, pred_cls = session.run([accuracy, label_batch, y_pred_cls])

            cls_wise_gt_count[gt_cls[0][0]] += 1
            true_positives[gt_cls[0][0]] += (gt_cls[0] == pred_cls[0])[:900]

            num_frames = acc.shape[1]
            for i in range(num_frames):
                if (i < 900):
                    acc_vs_frames_dict[i].append(acc[0][i])

            batch_counter += 1

            if batch_counter == test_batches_per_epoch:
                break

    true_positives /= cls_wise_gt_count.reshape([NUM_CLASSES, 1])
    true_positives *= 100

    print(true_positives[:, 0])
    print(true_positives[:, -1])

    for i in range(900):
        acc_vs_frames_dict[i] = 100 * float(sum(acc_vs_frames_dict[i])) / len(acc_vs_frames_dict[i])

    acc_vs_frames = np.zeros(900)
    for i in range(900):
        acc_vs_frames[i] = acc_vs_frames_dict[i]

    np.save(save_file, true_positives)
    # plt.plot(np.arange(1, 901, 1) / 60, acc_vs_frames)
    # np.save(save_file, acc_vs_frames)
    plt.xlabel('% of video observed')
    plt.ylabel('% accuracy')
    plt.show()



if __name__ == '__main__':
    lstm_test()





