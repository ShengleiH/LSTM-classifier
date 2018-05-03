import tensorflow as tf 
from tensorflow.contrib.rnn import LSTMCell
import numpy as np 


def last_relevant(output, length):

    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
  
    return relevant


class EGG_model(object):

    def __init__(self, input_dim, lstm_size, max_length, num_classes=3, learning_rate=0.001, num_layers=1, bidirectionoal=False):

        with tf.variable_scope('placeholders'):
            self.inputs = tf.placeholder(tf.float32, [None, max_length, input_dim], 'inputs')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')
            self.seq_lens = tf.placeholder(tf.int32, [None], 'seq_lens')
        
        with tf.variable_scope('lstm'):
            if not bidirectionoal:
                if num_layers > 1:
                    cell = tf.contrib.rnn.MultiRNNCell([LSTMCell(lstm_size) for _ in range(num_layers)]) 
                else:
                    cell = LSTMCell(lstm_size)
                # [batch_size, max_length, 513]
                outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, sequence_length=self.seq_lens, dtype=tf.float32)
                # [batch_size, 513]
                lasts = last_relevant(outputs, self.seq_lens)
            else:
                if num_layers > 1:
                    fw_cell = tf.contrib.rnn.MultiRNNCell([LSTMCell(lstm_size) for _ in range(num_layers)]) 
                    bw_cell = tf.contrib.rnn.MultiRNNCell([LSTMCell(lstm_size) for _ in range(num_layers)]) 
                else:
                    fw_cell = LSTMCell(lstm_size)
                    bw_cell = LSTMCell(lstm_size)
                # [batch_size, max_length, 513]
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.inputs, sequence_length=self.seq_lens, dtype=tf.float32)
                # [batch_size, max_length, 1026]
                cat_outputs = tf.concat(outputs, axis=2)
                # [batch_size, 1026] 
                lasts = last_relevant(cat_outputs, self.seq_lens)

        with tf.variable_scope('dense'):
            self.logits = tf.layers.dense(lasts, num_classes)  # B x 3
        
        with tf.variable_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
            self.loss = tf.reduce_mean(losses)
        
        with tf.variable_scope('optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
    def train(self, sess, feed_dict):
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
    
    def inference(self, sess, feed_dict):
        logits = sess.run(self.logits, feed_dict)
        return logits

