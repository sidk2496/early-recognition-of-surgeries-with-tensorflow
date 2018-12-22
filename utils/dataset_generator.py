import tensorflow as tf
import os


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def get_record_names(records_dir, split_name):
    path = os.path.join(records_dir, split_name)
    records_list = [os.path.join(path, record) for record in os.listdir(path)]
    return records_list


def _preprocess_image(image, split_name):

    if split_name == 'train':
        image = tf.random_crop(image, [224, 224, 3])
        # image = _distort_image(image)

    else:
        image = tf.image.crop_to_bounding_box(image, 14, 14, 224, 224)

    image = tf.subtract(image, IMAGENET_MEAN)
    image = image[:, :, ::-1]

    return image


def _distort_image(image):

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image


def _TFRecord_image_parser(example, split_name):
    features = {'image_data': tf.FixedLenFeature((), tf.string),
                'cls_label': tf.FixedLenFeature((), tf.int64),
                'frame': tf.FixedLenFeature((), tf.int64),
                'n_frames': tf.FixedLenFeature((), tf.int64),
                'video_id': tf.FixedLenFeature((), tf.string)}

    parsed_example = tf.parse_single_example(example, features)
    image = tf.decode_raw(parsed_example['image_data'], out_type=tf.uint8)
    image = tf.cast(tf.reshape(image, shape=[256, 256, 3]), tf.float32)
    image = _preprocess_image(image, split_name)
    label = parsed_example['cls_label']
    frame = parsed_example['frame']
    n_frames = parsed_example['n_frames']
    video_id = parsed_example['video_id']

    return image, label, frame, n_frames, video_id


def _TFRecord_feature_parser(example, feat_type):

    if feat_type in ['alex_fc7', 'vgg_16']:
        num_feat = 4096
    elif feat_type == 'endo_fc8':
        num_feat = 4103
    elif feat_type == 'res_50_blk4':
        num_feat = 2048
    elif feat_type == 'res_50_blk3':
        num_feat = 1024

    features = {'image_feat': tf.FixedLenFeature((num_feat), tf.float32),
                'cls_label': tf.FixedLenFeature((), tf.int64),
                'frame': tf.FixedLenFeature((), tf.int64),
                'n_frames': tf.FixedLenFeature((), tf.int64),
                'video_id': tf.FixedLenFeature((), tf.string)}

    parsed_example = tf.parse_single_example(example, features)
    feature = parsed_example['image_feat']
    label = parsed_example['cls_label']
    frame = parsed_example['frame']
    n_frames = parsed_example['n_frames']
    video_id = parsed_example['video_id']

    return feature, label, frame, n_frames, video_id


def _filter_initial_frames(image, label, frame, n_frames, video_id):
    return tf.not_equal(frame, 0)


class ImageDataset():

    def __init__(self, filenames, split_name, epoch_size=1, batch_size=1, type='TFRecord'):

        self._split_name = split_name
        self._type = type

        def _map_func(TFRecord_name):
            return tf.data.TFRecordDataset(TFRecord_name).\
                           map(self._parser, num_parallel_calls=16).\
                           filter(_filter_initial_frames)

        self._dataset = tf.data.Dataset.list_files(filenames).\
                                       shuffle(len(filenames)).\
                                       interleave(map_func=_map_func, cycle_length=20, block_length=1).\
                                       repeat(epoch_size). \
                                       shuffle(500).\
                                       batch(batch_size).\
                                       prefetch(1)

    def _parser(self, example):

        if self._type == 'TFRecord':
            return _TFRecord_image_parser(example, self._split_name)

    def get_dataset(self):
        return self._dataset


class ImageFeatureDataset():

    def __init__(self, filenames, split_name, epoch_size=1, batch_size=1, type='TFRecord', feat_type=None):
        self._split_name = split_name
        self._type = type
        self._feat_type = feat_type

        def _map_func(TFRecord_name):
            return tf.data.TFRecordDataset(TFRecord_name).\
                           map(self._parser, num_parallel_calls=16).\
                           filter(_filter_initial_frames)

        self._dataset = tf.data.Dataset.list_files(filenames).\
                                       shuffle(len(filenames)).\
                                       interleave(map_func=_map_func, cycle_length=20, block_length=1).\
                                       repeat(epoch_size). \
                                       shuffle(500).\
                                       batch(batch_size).\
                                       prefetch(1)

    def _parser(self, example):
        if self._type == 'TFRecord':
            return _TFRecord_feature_parser(example, self._feat_type)

    def get_dataset(self):
        return self._dataset



class VideoFramesDataset():

    def __init__(self, filenames, split_name, epoch_size=1, batch_size=1, type='TFRecord'):

        batch_size = batch_size
        self._split_name = split_name
        self._type = type
        epoch_size = epoch_size

        self._dataset = tf.data.Dataset.list_files(filenames).\
                                       shuffle(len(filenames)).\
                                       flat_map(tf.data.TFRecordDataset).\
                                       map(self._parser, num_parallel_calls=16).\
                                       batch(batch_size).\
                                       repeat(epoch_size).\
                                       prefetch(1)

    def _parser(self, example):

        if self._type == 'TFRecord':
            return _TFRecord_image_parser(example, self._split_name)

    def get_dataset(self):
        return self._dataset
    


class VideoFeatureDataset():

    def __init__(self, filenames, split_name, epoch_size=1, max_timestep=99999, batch_size=1, type='TFRecord',
                 feat_type=None):
        batch_size = batch_size
        self._split_name = split_name
        self._type = type
        epoch_size = epoch_size
        max_timestep = max_timestep
        self._feat_type = feat_type

        def _sub_batch(video_batch, label_batch, frame, n_frames, video_id):
            length = tf.cast(tf.shape(video_batch)[0], tf.float32)
            video_batch = video_batch[tf.cast(0.5 * length, tf.int64):]
            label_batch = label_batch[tf.cast(0.5 * length, tf.int64):]
            frame = frame[tf.cast(0.5 * length, tf.int64):]
            n_frames = n_frames[tf.cast(0.5 * length, tf.int64):]
            video_id = video_id[tf.cast(0.5 * length, tf.int64):]
            return video_batch, label_batch, frame, n_frames, video_id

        def _map_func(TFRecord_name):
            return tf.data.TFRecordDataset(TFRecord_name).\
                           map(self._parser, num_parallel_calls=16). \
                           filter(_filter_initial_frames).\
                           batch(max_timestep)
             

        self._dataset = tf.data.Dataset.list_files(filenames).\
                                       shuffle(len(filenames)).\
                                       flat_map(_map_func).\
                                       batch(batch_size).\
                                       repeat(epoch_size).\
                                       prefetch(1)

    def get_dataset(self):
        return self._dataset

    def _parser(self, example):

        if self._type == 'TFRecord':
            return _TFRecord_feature_parser(example, self._feat_type)
