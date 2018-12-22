import os
from utils.dataset_generator import get_record_names, VideoFramesDataset
from utils.TFRecord_utils import write_features_to_TFRecord
from model_build.cnn_build import *
from time import time
from datetime import timedelta

NUM_CLASSES = 9

batch_size = 256

gpu_id = 0

weights_path = 'my_weights_path'
data_records_dir = 'my_data_path'
output_dir = 'my_output_path'


def extract_features():

    train_records = get_record_names(data_records_dir, split_name='train')
    validation_records = get_record_names(data_records_dir, split_name='validation')
    test_records = get_record_names(data_records_dir, split_name='test')

    with tf.device('/device:CPU:0'):
        images = VideoFramesDataset(filenames=train_records[0],
                                    batch_size=batch_size,
                                    type='TFRecord',
                                    split_name='train').get_dataset()

        data_split_handle = tf.placeholder(dtype=tf.string)
        iterator = tf.data.Iterator.from_string_handle(data_split_handle, images.output_types, images.output_shapes)
        image_batch, label_batch, frame_batch, n_frames_batch, video_id_batch = iterator.get_next()



    with tf.device('/device:GPU:{0}'.format(gpu_id)):

        is_train = tf.placeholder(shape=(), dtype=tf.bool)
        model = VGGNet(inputs=image_batch, true_labels=label_batch, is_train=is_train, num_classes=NUM_CLASSES)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    init_op = tf.global_variables_initializer()
    
    errors = []

    with tf.Session(config=config) as session:
        
        session.run(init_op)
        model.load_initial_weights(weights_path, session)

        for records, split_name in zip([train_records, validation_records, test_records], ['train', 'validation', 'test']):
            for i, record in enumerate(records):
                start_time = time()
                images = VideoFramesDataset(filenames=record,
                                            batch_size=batch_size,
                                            type='TFRecord',
                                            split_name='test').get_dataset()
                iter = images.make_one_shot_iterator()
                handle = session.run(iter.string_handle())
                output_record_filename = 'vgg16_prog_' + record.split('/')[-1][:-9] + '.tfrecord'
                output_filepath = os.path.join(output_dir, split_name, output_record_filename)
                tfrecord_writer = tf.python_io.TFRecordWriter(output_filepath)
                while True:
                    try:
                        features, true_labels, frame, n_frames, video_id = session.run([model.extract_features('fc7'),
                                                                                        label_batch, frame_batch,
                                                                                        n_frames_batch, video_id_batch],
                                                                                        feed_dict={
                                                                                            data_split_handle: handle,
                                                                                            is_train: False})

                    except tf.errors.OutOfRangeError:
                        break

                    write_features_to_TFRecord({'image_feat': features, 'cls_label': true_labels, 'frame': frame,
                                                'n_frames': n_frames, 'video_id': video_id}, tfrecord_writer)
                print('Extracted features for {0:3d}/{1:3d} {2:>10} videos | Time taken: {3}'.
                      format(i + 1, len(records), split_name, timedelta(seconds=round(int(time() - start_time)))))

    for error in errors:
        print(error + '\n')

if __name__ == '__main__':
    extract_features()
