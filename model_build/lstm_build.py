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

            elif loss_type == 'linear_weighted_avg':

                y_true = tf.one_hot(indices=self.true_labels, depth=self._num_classes)
                fn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))

                fp_losses = -tf.reduce_sum((1 - y_true) * tf.log(1 - tf.nn.softmax(logits)), axis=2)
                # ASSUMING BATCH_SIZE of 1
                num_timesteps = tf.cast(tf.shape(logits)[1], tf.float32)
                timestep_weights = tf.range(1, num_timesteps + 1)
                fp_loss = tf.reduce_mean((timestep_weights * fp_losses
                                          / num_timesteps))
                loss = fp_loss + fn_loss

                y_pred_cls = tf.argmax(logits[:, -1, :], axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels[:, -1]), tf.float32),
                                          axis=0)

            elif loss_type == 'linear_weighted_avg_2':

                y_true = tf.one_hot(indices=self.true_labels, depth=self._num_classes)

                # ASSUMING BATCH_SIZE of 1
                num_timesteps = tf.cast(tf.shape(logits)[1], tf.float32)
                timestep_weights = tf.range(1, num_timesteps + 1)
                fn_loss = tf.reduce_mean(timestep_weights *
                                         tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
                                         / num_timesteps)

                fp_losses = -tf.reduce_sum((1 - y_true) * tf.log(1 - tf.nn.softmax(logits)), axis=2)
                fp_loss = tf.reduce_mean(fp_losses)
                loss = fp_loss + fn_loss

                y_pred_cls = tf.argmax(logits[:, -1, :], axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels[:, -1]), tf.float32),
                                          axis=0)

            elif loss_type == 'exp_weighted_avg':

                y_true = tf.one_hot(indices=self.true_labels, depth=self._num_classes)
                fn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))

                fp_losses = -tf.reduce_sum((1 - y_true) * tf.log(1 - tf.nn.softmax(logits)), axis=2)
                # ASSUMING BATCH_SIZE of 1
                num_timesteps = tf.cast(tf.shape(logits)[1], tf.float32)
                timestep_weights = tf.exp(tf.range(1, num_timesteps + 1) - num_timesteps)
                fp_loss = tf.reduce_mean((timestep_weights * fp_losses
                                          / num_timesteps))
                loss = fp_loss + fn_loss

                y_pred_cls = tf.argmax(logits[:, -1, :], axis=1)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels[:, -1]), tf.float32),
                                          axis=0)

            elif loss_type == 'exp_weighted_avg_2':

                y_true = tf.one_hot(indices=self.true_labels, depth=self._num_classes)

                # ASSUMING BATCH_SIZE of 1
                num_timesteps = tf.cast(tf.shape(logits)[1], tf.float32)
                timestep_weights = tf.exp(tf.range(1, num_timesteps + 1) - num_timesteps)
                fn_loss = tf.reduce_mean(timestep_weights *
                                          tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
                                         / num_timesteps)

                fp_losses = -tf.reduce_sum((1 - y_true) * tf.log(1 - tf.nn.softmax(logits)), axis=2)
                fp_loss = tf.reduce_mean(fp_losses)
                loss = fp_loss + fn_loss

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

        def _max(accum_val, conf):
            curr_max = tf.reduce_max([accum_val, conf], axis=0)
            return curr_max

        def _avg(accum_val, conf):
            curr_avg = tf.divide((accum_val[0] * accum_val[1]) + conf, accum_val[0] + 1)
            return (accum_val[0] + 1, curr_avg)

        def _weighted_avg(accum_val, conf):
            curr_wt_avg = tf.divide(accum_val[1] * accum_val[0], accum_val[0] + 1) + conf
            return (accum_val[0] + 1, curr_wt_avg)

        def _moving_avg(accum_val, conf):
            curr_mov_avg = 0.9 * accum_val + 0.1 * conf
            return curr_mov_avg

        if pred_type == 'last':
            conf_agg = confidence

        elif pred_type == 'max':
            conf_agg = tf.scan(_max, confidence, initializer=-tf.ones(shape=[self._num_classes]))

        elif pred_type == 'avg':
            _, conf_agg = tf.scan(_avg, confidence, initializer=(tf.constant(0.0),
                                                                 tf.zeros(shape=[self._num_classes])))

        elif pred_type == 'weighted_avg':
            _, conf_agg = tf.scan(_weighted_avg, confidence, initializer=(tf.constant(0.0),
                                                                          tf.zeros(shape=[self._num_classes])))

        elif pred_type == 'moving_avg':
            conf_agg = tf.scan(_moving_avg, confidence, initializer=tf.zeros(shape=[self._num_classes]))

        else:
            raise ValueError('\'pred_type\' can be one of \'last\',  \'last\', \'avg\', \'max\' and \'weighted_avg\'')

        y_pred_cls = tf.expand_dims(tf.argmax(conf_agg, axis=1), 0)
        accuracy = tf.cast(tf.equal(x=y_pred_cls, y=self.true_labels), tf.float32)

        return accuracy, y_pred_cls

    def extract_features(self):
        output, _ = tf.nn.dynamic_rnn(cell=self._cell, inputs=self.sequence, dtype=tf.float32)
        return tf.squeeze(output)
