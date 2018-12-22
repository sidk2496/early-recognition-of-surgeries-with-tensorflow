from utils.train_utils import *
import model_build.cnn_build as cnn
from utils.dataset_generator import ImageDataset, get_record_names
from datetime import timedelta
from sys import stdout
from time import time
from math import ceil
from decimal import Decimal

NUM_CLASSES = 9

train_batch_size = 64
validation_batch_size = 64

train_disp_freq = 400
validation_disp_freq = 400
snapshot_freq = 5

epochs = 1000

# PATH ARGUMENTS
tf.app.flags.DEFINE_string('data_records_dir', None, 'Directory of TFRecord files')
tf.app.flags.DEFINE_string('summary_dir', None, 'Path to summary directory')
tf.app.flags.DEFINE_string('saver_dir', None, 'Path to snapshots directory')
tf.app.flags.DEFINE_string('restore_path', None, 'Path to model.ckpt file for weight initialization')

# HPARAMS ARGUMENTS
tf.app.flags.DEFINE_string('lr_range', '-3_-3','range of learning rates to tune')
tf.app.flags.DEFINE_string('wd_range', '-3_-3', 'range of weight decays to tune')
tf.app.flags.DEFINE_string('dp_range', '0.5_0.5','range of dropout ratios to tune')
tf.app.flags.DEFINE_integer('num_hparam_searches', '1','number of random combinations of hparams to search over')

# GPU ID
tf.app.flags.DEFINE_integer('gpu_id', '0','ID of GPU to use')

ARGS = tf.app.flags.FLAGS

lr_range = [int(ARGS.lr_range.split('_')[0]), int(ARGS.lr_range.split('_')[1])]
wd_range = [int(ARGS.wd_range.split('_')[0]), int(ARGS.wd_range.split('_')[1])]
dropout_range = [float(ARGS.dp_range.split('_')[0]), float(ARGS.dp_range.split('_')[1])]
num_hparam_searches = ARGS.num_hparam_searches

gpu_id = ARGS.gpu_id

data_records_dir = ARGS.data_records_dir
summary_dir = ARGS.summary_dir
saver_dir = ARGS.saver_dir
restore_path = None #ARGS.restore_path

weights_path = 'my_weights_path'

def cnn_train():

    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    weight_decay = tf.placeholder(dtype=tf.float32, name='weight_decay')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    train_records = get_record_names(data_records_dir, split_name='train')
    validation_records = get_record_names(data_records_dir, split_name='validation')

    num_train_frames = 1339833
    num_validation_frames = 216975

    train_batches_per_epoch = ceil(num_train_frames / float(train_batch_size))
    validation_batches_per_epoch = ceil(num_validation_frames / float(validation_batch_size))

    with tf.device('/cpu:0'):
        train_images = ImageDataset(train_records, epoch_size=epochs, batch_size=train_batch_size,
                                          split_name='train').get_dataset()
        validation_images = ImageDataset(validation_records, epoch_size=epochs, batch_size=validation_batch_size,
                            split_name='validation').get_dataset()

        data_split_handle = tf.placeholder(dtype=tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(data_split_handle, train_images.output_types,
                                                       train_images.output_shapes)
        image_batch, label_batch, _, _, _ = iterator.get_next()
        train_iterator = train_images.make_initializable_iterator()
        validation_iterator = validation_images.make_initializable_iterator()

    with tf.device('/device:GPU:{0}'.format(gpu_id)):

        is_train = tf.placeholder(shape=(), dtype=tf.bool, name='is_train')
        model = cnn.VGGNet(inputs=image_batch, true_labels=label_batch, is_train=is_train, num_classes=NUM_CLASSES)
        accuracy, loss = model.compute_accuracy_and_loss(add_l2_reg=True, weight_decay=weight_decay)
        train_op = model.train(loss, lr=learning_rate)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    train_avg_accuracy = tf.Variable(dtype=tf.float32, trainable=False, initial_value=0.0, name='accuracy_train')
    train_avg_loss = tf.Variable(dtype=tf.float32, trainable=False, initial_value=0.0, name='loss_train')
    train_acc_summary = tf.summary.scalar('train_accuracy', train_avg_accuracy)
    train_loss_summary = tf.summary.scalar('train_loss', train_avg_loss)
    train_summary = tf.summary.merge([train_acc_summary, train_loss_summary])

    validation_avg_accuracy = tf.Variable(dtype=tf.float32, trainable=False, initial_value=0.0,
                                          name='accuracy_validation')
    validation_avg_loss = tf.Variable(dtype=tf.float32, trainable=False, initial_value=0.0, name='loss_validation')
    validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_avg_accuracy)
    validation_loss_summary = tf.summary.scalar('validation_loss', validation_avg_loss)
    validation_summary = tf.summary.merge([validation_acc_summary, validation_loss_summary])

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    with tf.Session(config=config) as session:

        for _ in range(num_hparam_searches):
            #============================================== hPARAMS ==================================================#

            hparam = get_rand_hparam(lr_range, wd_range, dropout_range)
            hparam_string = 'lr={0:.0E},wd={1:.0E},dp={2}'.format(Decimal(hparam['lr']),
                                                                       Decimal(hparam['wd']),
                                                                       hparam['dropout'])
            saver_subdir, summary_subdir = create_saver_summary_subdirs(saver_dir, summary_dir,
                                                                                    hparam_string)
            write_config_file(saver_subdir, hparam, 'SGD with Momentum', epochs, train_disp_freq,
                              validation_disp_freq,
                              train_batch_size, validation_batch_size)
            summary_writer = tf.summary.FileWriter(summary_subdir)

            #==============================================================================================================#

            session.run(init_op)
            if restore_path != None:
                saver.restore(session, restore_path)
            model.load_initial_weights(weights_path, session)

            summary_writer.add_graph(session.graph)

            session.run(train_iterator.initializer)
            session.run(validation_iterator.initializer)
            train_handle = session.run(train_iterator.string_handle())
            validation_handle = session.run(validation_iterator.string_handle())

            train_feed_dict = {learning_rate: hparam['lr'],
                               weight_decay: hparam['wd'],
                               keep_prob: 1 - hparam['dropout'],
                               data_split_handle: train_handle,
                               is_train: True}

            validation_feed_dict = {weight_decay: hparam['wd'],
                                    keep_prob: 1 - hparam['dropout'],
                                    data_split_handle: validation_handle,
                                    is_train: False}

            for epoch in range(epochs):

                #============================================== TRAINING ==================================================#

                if (epoch + 1) % 100 == 0:
                    train_feed_dict[learning_rate] = hparam['lr'] / 10

                avg_acc = []
                avg_loss = []
                batch_counter = 0
                start_time = time()

                while True:

                    # stdout.write('\r>>Training batch {0:3d}/{1:3d}....'.format(batch_counter + 1,
                    #                                                            train_batches_per_epoch))
                    stdout.flush()
                    _, acc, los = session.run([train_op, accuracy, loss], feed_dict=train_feed_dict)
                    avg_acc.append(acc)
                    avg_loss.append(los)
                    batch_counter += 1

                    if (batch_counter % train_disp_freq == 0) or (batch_counter == train_batches_per_epoch):
                        time_diff = timedelta(seconds=int(round(time() - start_time)))
                        iteration = epoch * train_batches_per_epoch + batch_counter
                        session.run(train_avg_accuracy.assign(value=100 * (sum(avg_acc) / len(avg_acc))))
                        session.run(train_avg_loss.assign(value=sum(avg_loss) / len(avg_loss)))

                        stdout.write('\rEPOCH: {0:3d}/{4:3d}   |   TRAIN   |   LOSS: {1:7.4f}   |   ACCURACY: {2:6.2f}'
                                     '   |   Time taken: {3}\n'.format(epoch + 1, session.run(train_avg_loss),
                                                                       session.run(train_avg_accuracy), time_diff,
                                                                       epochs))

                        summary = session.run(train_summary)
                        summary_writer.add_summary(summary, iteration)

                        avg_acc = []
                        avg_loss = []
                        start_time = time()

                    if batch_counter == train_batches_per_epoch:
                        break

                #============================================= VALIDATION =================================================#

                avg_acc = []
                avg_loss = []
                batch_counter = 0
                start_time = time()

                while True:

                    # stdout.write('\r>>Validating batch {0:3d}/{1:3d}....'.format(batch_counter + 1,
                    #                                                              validation_batches_per_epoch))
                    stdout.flush()
                    acc, los = session.run([accuracy, loss], feed_dict=validation_feed_dict)
                    avg_acc.append(acc)
                    avg_loss.append(los)
                    batch_counter += 1

                    if (batch_counter % validation_disp_freq == 0) or (batch_counter == validation_batches_per_epoch):
                        time_diff = timedelta(seconds=int(round(time() - start_time)))
                        iteration = epoch * validation_batches_per_epoch + batch_counter

                        session.run(validation_avg_accuracy.assign(value=100 * (sum(avg_acc) / len(avg_acc))))
                        session.run(validation_avg_loss.assign(value=sum(avg_loss) / len(avg_loss)))

                        stdout.write('\rEPOCH: {0:3d}/{4:3d}   |    VAL    |   LOSS: {1:7.4f}   |   ACCURACY: {2:6.2f}'
                                     '   |   Time taken: {3}\n'. format(epoch + 1, session.run(validation_avg_loss),
                                            session.run(validation_avg_accuracy),
                                            time_diff, epochs))

                        summary = session.run(validation_summary)
                        summary_writer.add_summary(summary, iteration)

                        avg_acc = []
                        avg_loss = []
                        start_time = time()

                    if batch_counter == validation_batches_per_epoch:
                        break

                #==========================================================================================================#

                stdout.write('\n')

                if (epoch + 1) % snapshot_freq == 0:
                    save_path = saver_subdir + '/model_{0:04d}.ckpt'.format(epoch + 1)
                    saver.save(sess=session, save_path=save_path)


if __name__ == '__main__':
    cnn_train()

