import numpy as np
import tensorflow as tf
import model_build.utils.cnn_utils as cnn
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import vgg


class AlexNet():

    def __init__(self, input, true_labels, num_classes, keep_prob=1.0, is_train=True):

        self.input = input
        self.true_labels = true_labels
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAIN = is_train

        self.fc7 = None
        self.fc8 = None

        self.create()

    def create(self):

        conv1, _ = cnn.conv(self.input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = cnn.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = cnn.lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        conv2, _ = cnn.conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = cnn.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = cnn.lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        conv3, _ = cnn.conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        conv4, _ = cnn.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        conv5, _ = cnn.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = cnn.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        flattened = cnn.flatten(pool5)
        fc6, _ = cnn.fc(flattened, 4096, name='fc6')

        dropout6 = tf.cond(self.IS_TRAIN, lambda: cnn.dropout(fc6, self.KEEP_PROB, name='dropout6'),
                           lambda: fc6)

        self.fc7, _ = cnn.fc(dropout6, 4096, name='fc7')
        dropout7 = tf.cond(self.IS_TRAIN, lambda: cnn.dropout(self.fc7, self.KEEP_PROB, name='dropout7'),
                           lambda: self.fc7)

        self.fc8, _ = cnn.fc(dropout7, self.NUM_CLASSES, use_relu=False, name='fc8')

    def load_initial_weights(self, weights_path, session):

        weights_dict = np.load(weights_path, encoding='bytes').item()

        for op_name in weights_dict.keys():

            if op_name == 'fc8':
                continue

            with tf.variable_scope(op_name, reuse=True):

                for pre_trained_weights in weights_dict[op_name]:

                    if len(pre_trained_weights.shape) == 1:
                        var = tf.get_variable('biases')
                        session.run(var.assign(pre_trained_weights))

                    else:
                        var = tf.get_variable('weights')
                        session.run(var.assign(pre_trained_weights))

    def extract_fc7(self):
        return self.fc7

    def extract_fc8(self):
        return self.fc8

    def compute_accuracy_and_loss(self, add_l2_reg=False, weight_decay=0.001):

        y_true_one_hot = tf.one_hot(indices=self.true_labels, depth=self.NUM_CLASSES)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_one_hot,
                                                                      logits=self.fc8), axis=0)
        y_pred_cls = tf.argmax(self.fc8, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.true_labels, y_pred_cls), tf.float32), axis=0)
        if add_l2_reg:
            loss += weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        return accuracy, loss

    def train(self, loss, lr=0.001, momentum=0.9):
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(loss)
        return train_op



class EndoNet():

    def __init__(self, input, true_labels, num_classes, keep_prob=1.0, is_train=True):

        self.input = input
        self.true_labels = true_labels
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.IS_TRAIN = is_train

        self.fc8_concat = None

        self.create()

    def create(self):

        conv1, _ = cnn.conv(self.input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = cnn.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = cnn.lrn(pool1, 5, 1e-04, 0.75, name='norm1')

        conv2, _ = cnn.conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = cnn.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = cnn.lrn(pool2, 5, 1e-04, 0.75, name='norm2')

        conv3, _ = cnn.conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        conv4, _ = cnn.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        conv5, _ = cnn.conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = cnn.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        flattened = cnn.flatten(pool5)
        fc6, _ = cnn.fc(flattened, 4096, name='fc6')

        dropout6 = tf.cond(self.IS_TRAIN, lambda: cnn.dropout(fc6, self.KEEP_PROB, name='dropout6'),
                           lambda: fc6)

        fc7, _ = cnn.fc(dropout6, 4096, name='fc7')
        dropout7 = tf.cond(self.IS_TRAIN, lambda: cnn.dropout(fc7, self.KEEP_PROB, name='dropout7'),
                           lambda: fc7)

        fc8_clipper, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_clipper')
        fc8_hook, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_hook')
        fc8_grasperElec, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_grasperElec')
        fc8_scissorsElec, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_scissorsElec')
        fc8_grasperFen, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_grasperFen')
        fc8_irrigator, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_irrigator')
        fc8_specBag, _ = cnn.fc(dropout7, 1, use_relu=False, name='fc8_specBag')

        self.fc8_concat = tf.concat([fc7, fc8_clipper, fc8_hook,
                                         fc8_grasperFen, fc8_grasperElec, fc8_scissorsElec, fc8_specBag,
                                         fc8_irrigator], axis=1)


    def load_initial_weights(self, weights_path, session):

        weights_dict = np.load(weights_path, encoding='bytes').item()

        for op_name in weights_dict.keys():

            if op_name == 'fc8_surgery':
                continue

            with tf.variable_scope(op_name, reuse=True):
                for key in weights_dict[op_name].keys():

                    if key == b'weights':
                        var = tf.get_variable('weights')
                        if len(weights_dict[op_name][key].shape) == 1:
                            weights_dict[op_name][key] = np.expand_dims(weights_dict[op_name][key], axis=1)
                        session.run(var.assign(weights_dict[op_name][key]))

                    else:
                        var = tf.get_variable('biases')
                        if len(weights_dict[op_name][key].shape) == 0:
                            weights_dict[op_name][key] = np.expand_dims(weights_dict[op_name][key], axis=0)
                        session.run(var.assign(weights_dict[op_name][key]))

    def extract_fc8(self):
        return self.fc8_concat



class ResNet():

    def __init__(self, inputs, true_labels, is_train=False, num_classes=None):

        self.true_labels = true_labels
        self.NUM_CLASSES = num_classes
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            self.output, self.features = resnet_v2.resnet_v2_50(inputs=inputs, num_classes=None, is_training=False)
            self.classifier, _ = cnn.fc(input=self.forward_pass(), num_outputs=self.NUM_CLASSES, use_relu=False, name='classifier')


    def load_initial_weights(self, weights_path, session):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
        saver.restore(session, weights_path)


    def forward_pass(self):
        return tf.reshape(self.output, [-1, 2048])

    def extract_features(self, layer):
        return tf.reduce_mean(tf.reduce_mean(self.features['resnet_v2_50/' + layer], axis=2), axis=1)


    def compute_accuracy_and_loss(self, add_l2_reg=False, weight_decay=0.001):

        y_true_one_hot = tf.one_hot(indices=self.true_labels, depth=self.NUM_CLASSES)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_one_hot,
                                                                      logits=self.classifier), axis=0)
        y_pred_cls = tf.argmax(self.classifier, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.true_labels, y_pred_cls), tf.float32), axis=0)
        if add_l2_reg:
            loss += weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        return accuracy, loss


    def train(self, loss, lr=0.001, momentum=0.9):
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(loss)
        return train_op


class VGGNet():

    def __init__(self, inputs, true_labels, is_train=False, num_classes=None):

        self.true_labels = true_labels
        self.NUM_CLASSES = num_classes
        with slim.arg_scope(vgg.vgg_arg_scope()):
            self.output, self.features = vgg.vgg_16(inputs=inputs, num_classes=1000, is_training=False)
            self.classifier, _ = cnn.fc(input=self.extract_features('fc7'), num_outputs=self.NUM_CLASSES, use_relu=False,
                                        name='classifier')

    def load_initial_weights(self, weights_path, session):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16'))
        saver.restore(session, weights_path)

    def extract_features(self, layer):
        return tf.reduce_mean(tf.reduce_mean(self.features['vgg_16/' + layer], axis=2), axis=1)

    def forward_pass(self):
        return self.output

    def compute_accuracy_and_loss(self, add_l2_reg=False, weight_decay=0.001):

        y_true_one_hot = tf.one_hot(indices=self.true_labels, depth=self.NUM_CLASSES)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true_one_hot,
                                                                      logits=self.classifier), axis=0)
        y_pred_cls = tf.argmax(self.classifier, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.true_labels, y_pred_cls), tf.float32), axis=0)
        if add_l2_reg:
            loss += weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        return accuracy, loss

    def compute_loss_for_progress(self, add_l2_reg=False, weight_decay=0.001):
        prog_pred= tf.sigmoid(cnn.fc(self.extract_features('fc7'), num_outputs=1, use_relu=False, name='fc_prog')[0])
        loss = tf.losses.huber_loss(self.true_labels, tf.squeeze(prog_pred))
        if add_l2_reg:
            loss += weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        return loss

    def train(self, loss, lr=0.001, momentum=0.9):
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum).minimize(loss)
        return train_op

    def test(self):
        y_pred_cls = tf.argmax(self.classifier, axis=1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(self.true_labels, y_pred_cls), tf.float32), axis=0)
        return accuracy
