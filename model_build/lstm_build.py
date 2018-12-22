import tensorflow as tf

class LSTM(object):

    def __init__(self, input, true_labels, num_hidden, num_classes, name, use_dropout=False,
                 input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0):

        self._name = name
        self.sequence = input
        self.true_labels = true_labels
        with tf.name_scope(name):
            self._num_hidden = num_hidden
            self._cell = tf.nn.rnn_cell.LSTMCell(self._num_hidden)
            if use_dropout:
                self._cell = tf.nn.rnn_cell.DropoutWrapper(self._cell, input_keep_prob,
                                                           output_keep_prob, state_keep_prob)
            self._num_classes = num_classes
            # self.initial_state = self._cell.zero_state(batch_size=, dtype=tf.float32)
            self._softmax_w = tf.get_variable('softmax_w',
                                              initializer=tf.truncated_normal([self._num_hidden, self._num_classes],
                                                                              stddev=0.05))
            self._softmax_b = tf.get_variable('softmax_b',
                                              initializer=tf.constant(0.05, shape=[self._num_classes]))

    def load_initial_weights(self, session, weights_path):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(session, weights_path)

    def compute_accuracy_and_loss(self, add_l2_reg=False, weight_decay=0, loss_type=None):

        with tf.name_scope(self._name):
            output, _ = tf.nn.dynamic_rnn(cell=self._cell, inputs=self.sequence, dtype=tf.float32)
            logits = tf.map_fn(lambda x: tf.matmul(x, self._softmax_w) + self._softmax_b, output)

            if loss_type == 'last':

                y_true = tf.one_hot(indices=self.true_labels[:, 0], depth=self._num_classes)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, -1, :], labels=y_true),
                                      axis=0)

                y_pred_cls = tf.argmax(logits[:, -1, :], axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels[:, -1]), tf.float32),
                                          axis=0)

            elif loss_type == 'avg':

                y_true = tf.one_hot(indices=self.true_labels, depth=self._num_classes)
                # NOTE:- We are able to use y_true[0] is used because batch size is 1.
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=y_true))

                y_pred_cls = tf.argmax(logits[:, -1, :], axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels[:, -1]), tf.float32),
                                          axis=0)

            if add_l2_reg:
                loss += weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        return accuracy, loss

    def train(self, loss, lr, momentum=0.9):
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
        train_op = optimizer.minimize(loss)
        return  train_op

    def test(self, pred_type='last'):

        output, _ = tf.nn.dynamic_rnn(cell=self._cell, inputs=self.sequence, dtype=tf.float32)
        logits = tf.map_fn(lambda x: tf.matmul(x, self._softmax_w) + self._softmax_b, output)
        # NOTE:- Batch size has to be 1.
        confidence = tf.reshape(tf.nn.softmax(logits), [-1, self._num_classes])

        if pred_type == 'last':
            conf_agg = confidence

        y_pred_cls = tf.expand_dims(tf.argmax(conf_agg, axis=1), 0)
        accuracy = tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels), tf.float32)

        return accuracy, y_pred_cls

    def extract_features(self):
        output, _ = tf.nn.dynamic_rnn(cell=self._cell, inputs=self.sequence, dtype=tf.float32)
        return tf.squeeze(output)
