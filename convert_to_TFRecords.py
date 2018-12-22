import tensorflow as tf
from utils.TFRecord_utils import convert_video_dataset, get_splitwise_video_names_and_classes

#===============DEFINE YOUR ARGUMENTS==============
flags = tf.app.flags

# State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is
# essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.05, 'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_float('test_size', 0.40, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards per dataset split.
flags.DEFINE_integer('num_per_shard', 1024, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

# Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecords_path', None,
                    'String: The output filename to name your TFRecord file')

flags.DEFINE_string('video_filenames_dir', None, 'String: The path to the directory containing text files that have videos of different'
                    ' classes sorted in ascending order of their durations')


FLAGS = flags.FLAGS


def main():

    # Divide the training datasets into train and test:
    train_video_filenames, validation_video_filenames, test_video_filenames, class_names = \
                                                                 get_splitwise_video_names_and_classes(
                                                                 video_filenames_dir=FLAGS.video_filenames_dir,
                                                                 tfrecords_path = FLAGS.tfrecords_path,
                                                                 validation_size=FLAGS.validation_size,
                                                                 test_size=FLAGS.test_size)
    
    print('Number of TRAIN videos: ', len(train_video_filenames),
           '\nNumber of VALIDATION videos: ', len(validation_video_filenames),
           '\nNumber of TEST videos: ', len(test_video_filenames))
    
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    
    convert_video_dataset('train', train_video_filenames, FLAGS.dataset_dir, class_names_to_ids, FLAGS.tfrecords_path
                          , decode_png=True)

    convert_video_dataset('validation', validation_video_filenames, FLAGS.dataset_dir, class_names_to_ids,
                          FLAGS.tfrecords_path, decode_png=True)

    convert_video_dataset('test', test_video_filenames, FLAGS.dataset_dir, class_names_to_ids, FLAGS.tfrecords_path,
                          decode_png=True)

    print('\nFinished converting {0} dataset!'.format(FLAGS.tfrecords_path))

if __name__ == "__main__":
    main()
