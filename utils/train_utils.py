import tensorflow as tf
import os
from subprocess import Popen
from numpy import arange
from os.path import join
from random import randint, choice


def get_rand_hparam(lr_range, wd_range, dropout_range):

    hyper_param = {}
    # hyper_param['lstm_dim'] = 2 ** randint(log2(lstm_dim_range[0]), log2(lstm_dim_range[1]))
    hyper_param['lr'] = 10 ** randint(lr_range[0], lr_range[1])
    hyper_param['wd'] = 10 ** randint(wd_range[0], wd_range[1])
    hyper_param['dropout'] = choice(arange(dropout_range[0], dropout_range[1] + 0.1, 0.1))

    return hyper_param


def write_config_file(save_path, hyper_param, optimizer_name, epochs, train_disp_freq, validation_disp_freq,
                      train_batch_size, validation_batch_size, lstm_dim=None):

    with open(join(save_path, 'config.txt'), 'w') as config_file:

        config_file.write('lstm_dim: {0}\nlearning rate: {1}\nmomentum: {2}\noptimizer: {3}\nwd: {4}\nepochs:'
                          ' {5}\ntrain_disp_freq: {6}\nvalidation_disp_freq: {7}\ntrain_batch_size : {8}\n'
                          'validation_batch_size: {9}'.format(lstm_dim, hyper_param['lr'], 0.9,
                                                              optimizer_name, hyper_param['wd'], epochs,
                                                              train_disp_freq, validation_disp_freq, train_batch_size,
                                                              validation_batch_size))


def create_saver_summary_subdirs(saver_dir, summary_dir, hyper_param_string):

    saver_subdir = join(saver_dir, hyper_param_string)
    summary_subdir = join(summary_dir, hyper_param_string)
    Popen(['mkdir', saver_subdir])
    Popen(['mkdir', summary_subdir])
    return saver_subdir, summary_subdir


def get_total_num_frames(video_frames_dir, tfrecords, reject_start=True):
    num_images = 0

    if reject_start:
        with open('path_to_video_list.txt'.format(tfrecords[0].split('/')[-2]), 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                num_images += ((int(line.split(';')[-1]) / 25 - int(line.split(';')[-2]) / 25 + 1) // 100) * 100
        return num_images

    else:
        video_frames_subdirs = [os.path.join(video_frames_dir, class_name) for class_name in os.listdir(video_frames_dir)]
        for tfrecord in tfrecords:
            video_name = tfrecord.split('/')[-1][:-9]
            for dir in video_frames_subdirs:
                if video_name in os.listdir(dir):
                    num_images += len(os.listdir(os.path.join(dir, video_name)))
                    break

    return num_images
