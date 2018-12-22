import tensorflow as tf
from math import ceil, floor
import os
import sys
import numpy as np
from random import sample


# Wrapper functions for creating Example features

def float32_feature(values):
    if not isinstance(values, (np.ndarray, tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def int64_feature(values):
    if not isinstance(values, (np.ndarray, tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


# Utility functions for converting an image to Example

def image_to_tfexample(image_data, cls_label, frame, n_frames, video_id, decode_png):
    '''This function converts a png image to an example WITHOUT decoding it.'''

    if decode_png:
        image_data = image_data.tostring()
        return tf.train.Example(features=tf.train.Features(feature={
            'image_data': bytes_feature(image_data),
            'cls_label': int64_feature(cls_label),
            'frame': int64_feature(frame),
            'n_frames': int64_feature(n_frames),
            'video_id': bytes_feature(video_id)}))

    else:
        return tf.train.Example(features=tf.train.Features(feature={
        'image_data': bytes_feature(image_data),
        'cls_label': int64_feature(cls_label),
        'frame': int64_feature(frame),
        'n_frames': int64_feature(n_frames),
        'video_id': bytes_feature(video_id)}))


def features_to_example(features, cls_label, frame, n_frames, video_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image_feat': float32_feature(values=features),
        'cls_label': int64_feature(cls_label),
        'frame': int64_feature(frame),
        'n_frames': int64_feature(n_frames),
        'video_id': bytes_feature(video_id)}))


def create_decode_png_graph(image_bytes):
    image = tf.image.decode_png(contents=image_bytes, channels=3)
    image = tf.image.resize_images(image, [256, 456], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.crop_to_bounding_box(image, 0, 70, 256, 310)
    image = tf.image.resize_images(img, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(tf.reshape(image, shape=[-1]), tf.uint8)
    return image


def write_image_to_TFRecord(session, tfrecord_writer, image_decoder, image_holder, image_path, cls_label=0, frame=1,
                            n_frames=1, video_id='NA', decode_png=True):

    image_bytes = tf.gfile.FastGFile(image_path, 'rb').read()
    if decode_png:
        image_data = session.run(image_decoder, feed_dict={image_holder: image_bytes})
        example = image_to_tfexample(image_data=image_data, cls_label=cls_label, frame=frame, n_frames=n_frames,
                                     video_id=video_id, decode_png=decode_png)
    else:
        example = image_to_tfexample(image_data=image_bytes, cls_label=cls_label, frame=frame, n_frames=n_frames,
                                     video_id=video_id, decode_png=decode_png)
    tfrecord_writer.write(example.SerializeToString())


def write_image_dir_to_TFRecord(session, tfrecords_path, image_dir, cls_label, decode_png, start_frame=1, end_frame=1,
                                count_video=1, num_videos=1):
    '''This function writes all images in a directory to a TFRecord.'''

    image_holder = tf.placeholder(dtype=tf.string)
    image_decoder = create_decode_png_graph(image_holder)
    video_name = image_dir.split('/')[-1]
    video_id = video_name.split('H')[0].encode()
    tfrecord_file = '{0}.tfrecord'.format(os.path.join(tfrecords_path, video_name))
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file)
    image_files = os.listdir(image_dir)
    image_files.sort()
    num_pre_surgery_frames = len([img for img in image_files if int(img.split('.')[0]) < start_frame])
    num_images = len(image_files) - num_pre_surgery_frames

    for count, image_file in enumerate(image_files):

        frame = 0 if count < num_pre_surgery_frames else (count + 1 - num_pre_surgery_frames)
        sys.stdout.write('\rWriting image {0}/{1} of video {2}/{3}....'.
                         format(count + 1, len(image_files), count_video, num_videos))
        sys.stdout.flush()
        image_path = os.path.join(image_dir, image_file)
        write_image_to_TFRecord(session=session, tfrecord_writer=tfrecord_writer, image_decoder=image_decoder,
                                image_holder=image_holder, image_path=image_path, cls_label=cls_label,
                                frame=frame, n_frames=num_images,
                                video_id=video_id, decode_png=decode_png)


def _generate_split_indices(num_points, validation_size, test_size):
    '''This function returns indices for train, validation and test from indices in the range 0 to num_points.'''

    quart_size = num_points // 4
    quartiles = [[]] * 4
    for i in range(4):
        if i == 3:
            quartiles[i] = range(i * quart_size, num_points)
        else:
            quartiles[i] = range(i * quart_size, (i + 1) * quart_size)

    train_indices = []
    validation_indices = []
    test_indices = []
    for i in range(4):
        indices = sample(quartiles[i], floor(test_size * quart_size))
        test_indices += indices
        quartiles[i] = [idx for idx in quartiles[i] if idx not in indices]

    for i in range(4):
        indices = sample(quartiles[i], ceil(validation_size * quart_size))
        validation_indices += indices
        quartiles[i] = [idx for idx in quartiles[i] if idx not in indices]

    for i in range(4):
        train_indices += quartiles[i]

    return train_indices, validation_indices, test_indices


def _time_in_sec(time):
    hr, min, sec = time.split(':')
    return (int(hr) * 3600) + (int(min) * 60) + int(sec)


def _sort_by_duration(line):
    start_time, end_time = line.split(';')[1: ]
    duration = _time_in_sec(end_time) - _time_in_sec(start_time)
    return duration


def get_splitwise_video_names_and_classes(video_filenames_dir, tfrecords_path, validation_size, test_size):
    '''This function returns video names for train, validation and test splits.'''

    train_video_filenames = []
    validation_video_filenames = []
    test_video_filenames = []
    class_names = []

    for filename in os.listdir(video_filenames_dir):
        path = os.path.join(video_filenames_dir, filename)
        class_name = filename.split('.')[0]
        class_names.append(class_name)

        with open(path, 'r') as file:
            lines = file.readlines()
            lines.sort(key=_sort_by_duration)
            train_indices, validation_indices, test_indices = _generate_split_indices(len(lines), validation_size,
                                                                                      test_size)
            for line in [lines[i] for i in train_indices]:
                train_video_filenames.append({'name': os.path.join(class_name, line.split(';')[0].split('/')[-1]),
                                             'start_frame': _time_in_sec(line.split(';')[1]) * 25,
                                             'end_frame': _time_in_sec(line.split(';')[2]) * 25})

            for line in [lines[i] for i in validation_indices]:
                validation_video_filenames.append({'name': os.path.join(class_name, line.split(';')[0].split('/')[-1]),
                                                   'start_frame': _time_in_sec(line.split(';')[1]) * 25,
                                                   'end_frame': _time_in_sec(line.split(';')[2]) * 25})

            for line in [lines[i] for i in test_indices]:
                test_video_filenames.append({'name': os.path.join(class_name, line.split(';')[0].split('/')[-1]),
                                             'start_frame': _time_in_sec(line.split(';')[1]) * 25,
                                             'end_frame': _time_in_sec(line.split(';')[2]) * 25})

    # video names are in the format 'CLASS_NAME/VIDEO_NAME'
    for split_name, videos in zip(['train', 'validation', 'test'], [train_video_filenames,
                                                                         validation_video_filenames,
                                                                         test_video_filenames]):
        with open(os.path.join(tfrecords_path, '{0}_videos_list.txt'.format(split_name)), 'w') as file:
            for video in videos:
                file.write('{0};{1};{2}\n'.format(video['name'], video['start_frame'], video['end_frame']))

    return train_video_filenames, validation_video_filenames, test_video_filenames, sorted(class_names)


def convert_video_dataset(split_name, videos, video_frames_dir, class_names_to_ids, tfrecords_path,
                          decode_png=True):

    tfrecords_path = os.path.join(tfrecords_path, split_name)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        for count_video, video in enumerate(videos):
            class_name = video['name'].split('/')[0]
            image_dir = os.path.join(video_frames_dir, video['name'])
            cls_label = class_names_to_ids[class_name]
            write_image_dir_to_TFRecord(session, tfrecords_path, image_dir, cls_label, decode_png,
                                        video['start_frame'], video['end_frame'], count_video + 1, len(videos))

def write_features_to_TFRecord(feat_dict, tfrecord_writer):
    for i in range(feat_dict['image_feat'].shape[0]):
        example = features_to_example(feat_dict['image_feat'][i], feat_dict['cls_label'][i], feat_dict['frame'][i],
                                      feat_dict['n_frames'][i], feat_dict['video_id'][i])
        tfrecord_writer.write(example.SerializeToString())


# def video_to_sequence_example(dataset_dir, video_name, cls_label):
#
#     example = tf.train.SequenceExample()
#     video_path = os.path.join(dataset_dir, video_name)
#     video_frame_names = os.listdir(video_path)
#     sequences_length = len(video_frame_names)
#     example.context.feature["length"].int64_list.value.append(sequences_length)
#     fl_tokens = example.feature_lists.feature_list["tokens"]
#     fl_labels = example.feature_lists.feature_list["labels"]
#     labels = [cls_label for _ in range(sequences_length)]
#
#     for frame, label in zip(video_frame_names, labels):
#         token = open(os.path.join(video_path, frame), 'rb').read()
#         fl_tokens.feature.add().bytes_list.value.append(token)
#         fl_labels.feature.add().int64_list.value.append(label)
#
#     return example



