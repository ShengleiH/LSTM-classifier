import tensorflow as tf 
import numpy as np 
from model import EGG_model
from data_helper import EGGDataset
from scipy.stats import mode

max_steps = 100000
batch_size = -1
max_length = 50
input_dim = 310
lstm_size = 200
num_classes = 3
learning_rate = 0.0001
subject_ids=[0, 1, 2]
num_layers = 2
bidirectionoal=True

segment_value = 10
split_mode = 'segment'


def run():
    dataset = EGGDataset()
    dataset.read_data()  # read original data: data, lengths, max_lengths, labels
    train_X, train_y, train_lens, _, test_X, test_y, test_lens, test_batches \
                        = dataset.split(mode='segment', time_steps=max_length, subject_ids=subject_ids)    
    
    model = EGG_model(input_dim, lstm_size, max_length, num_classes, learning_rate, num_layers, bidirectionoal)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train
    for step in range(max_steps):
        batch_train_X, batch_train_y, batch_train_lens = dataset.get_batch(train_X, train_y, train_lens, batch_size)

        feed_dict = {
            model.inputs: batch_train_X,
            model.targets: batch_train_y,
            model.seq_lens: batch_train_lens
        }

        loss = model.train(sess, feed_dict) 

        # test
        feed_dict_test = {
            model.inputs: test_X,
            model.seq_lens: test_lens
        }

        logits = model.inference(sess, feed_dict_test)
        preds = np.argmax(logits, axis=1)
        same = 0
        assert len(test_batches) == 6 * 3
        for i, n_seg in enumerate(test_batches):
            seg_preds = preds[i * n_seg: (i+1) * n_seg]
            pred = mode(seg_preds)[0][0]
            seg_tgts = test_y[i * n_seg: (i+1) * n_seg]
            tgt = mode(seg_tgts)[0][0]
            if pred == tgt:
                same += 1
        
        assert len(test_batches) == 3 * 6
        accurancy = same / len(test_batches)

        print('step {}\tloss = {}\taccuracy = {}'.format(step, loss, accurancy))
        

if __name__ == '__main__':
    run()