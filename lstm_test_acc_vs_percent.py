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


NUM_CLASSES = 9

test_batch_size = 1
max_timestep = 20000

# PATH ARGUMENTS
tf.app.flags.DEFINE_string('data_records_dir', None, 'Directory of TFRecord files')
tf.app.flags.DEFINE_string('output_dir', None, 'Path to save results')
tf.app.flags.DEFINE_string('restore_path', None,'Path to model.ckpt file for weight initialization')
# GPU ID
tf.app.flags.DEFINE_integer('gpu_id', '0','ID of GPU to use')
# PREDICTION AGGREGATION TYPE
tf.app.flags.DEFINE_string('pred_type', 'last', 'Prediction aggregation scheme: can be any one of \'last\', \'max\', '
                                                '\'avg\' and \'weighted_avg\'')

ARGS = tf.app.flags.FLAGS

gpu_id = ARGS.gpu_id
pred_type = ARGS.pred_type

data_records_dir = ARGS.data_records_dir
restore_path = ARGS.restore_path
output_dir = ARGS.output_dir

tokens = restore_path.split('/')
save_file = os.path.join(output_dir, '{0}_{1}_{2}_{3}.npy'.format(pred_type, tokens[-4], tokens[-3], tokens[-2]))

def lstm_test():

    test_records = get_record_names(data_records_dir, split_name='test')
    test_batches_per_epoch = ceil(len(test_records) / float(test_batch_size))
    test_disp_freq = len(test_records)

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
    acc_vs_frames = np.zeros(100)
    conf_mat = np.zeros([9, 9], dtype=np.int32)

    with tf.Session(config=config) as session:

        saver.restore(session, restore_path)

        avg_acc = []
        batch_counter = 0
        start_time = time()
        while True:
            stdout.write('\r>>Testing batch {0:3d}/{1:3d}....'.format(batch_counter + 1,
                                                                       test_batches_per_epoch))
            stdout.flush()
            acc, gt_cls, pred_cls = session.run([accuracy, label_batch, y_pred_cls])
            num_frames = acc.shape[1]
            conf_mat[gt_cls[0, int(0.7 * num_frames)]][pred_cls[0, int(0.7 * num_frames)]] += 1
            acc_vs_frames += acc[0, (np.arange(0.01, 1.01, 0.01) * num_frames).astype(np.int64) - 1]
            avg_acc.append(acc[0, -1])
            batch_counter += 1
            if (batch_counter % test_disp_freq == 0) or (batch_counter == test_batches_per_epoch):
                time_diff = timedelta(seconds=int(round(time() - start_time)))
                stdout.write('\rACCURACY: {0:6.2f}   |   Time taken: {1}\n'.format(100 * (sum(avg_acc) / len(avg_acc)),
                                                                                   time_diff))
                avg_acc = []
                start_time = time()

            if batch_counter == test_batches_per_epoch:
                break


    acc_vs_frames /= len(test_records)
    acc_vs_frames *= 100

    plt.plot(np.arange(1, 101, 1), acc_vs_frames)
    np.save(save_file, acc_vs_frames)
    plt.xlabel('% of video observed')
    plt.ylabel('% accuracy')
    plt.show()

if __name__ == '__main__':
    lstm_test()





