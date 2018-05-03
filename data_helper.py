import numpy as np


def pad_seq(seqs, max_length):
        seqs_pad = []

        for seq in seqs:
            seq_pad = np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), 'constant', constant_values=0)
            assert seq_pad.shape == (max_length, seq.shape[1])
            seqs_pad.append(seq_pad)
        
        return seqs_pad


def segment(X, y, lens, time_steps=10):
    """
    Args: 
        X: n_samples * (real_length, 310)
        y: (n_samples, )
        lens: (n_samples, )
    
    Returns:
        segment_X: (n_samples * time_steps, 265 // time_steps, 310)
        segment_y: (n_samples * time_steps)
        segment_lens: (n_samples * time_steps)
    """
    segment_X = []
    segment_y = []
    segment_lens = []
    num_batches = []

    n_samples = len(segment_X)

    for x, label, length in zip(X, y, lens):
        old_time_steps = x.shape[0]
        n_segments = int(np.ceil(old_time_steps / time_steps))  # 235 / 20 = 12; 265 / 20 = 14
        num_batches.append(n_segments)
        
        for seg_idx in range(n_segments):
            if seg_idx == n_segments - 1:
                seg_x = x[-1 * time_steps:, :]  # time_step x 310
            else:
                seg_x = x[seg_idx * time_steps: (seg_idx + 1) * time_steps, :]
        
            segment_X.append(seg_x)
            segment_y.append(label)
            segment_lens.append(time_steps)

    segment_X = np.stack(segment_X)
    segment_y = np.stack(segment_y)
    segment_lens = np.stack(segment_lens)

    return segment_X, segment_y, segment_lens, num_batches


class EGGDataset(object):
    def __init__(self, data_dir='data_used/'):
        self.data_dir = data_dir

    def read_data(self):
        data = []
        lengths = []
        max_lengths = []
        labels = []

        sub_labels = np.load(self.data_dir + 'label.npy')

        for subjuct_id in range(1, 4):
            zip_data = np.load(self.data_dir + '0{}.npz'.format(subjuct_id))
            film_ids = zip_data.keys()

            sub_data = []
            for film_id in film_ids:
                film = zip_data[film_id]
                film = np.reshape(film, [-1, 310])
                sub_data.append(film)

            sub_lengths = [sd.shape[0] for sd in sub_data]
            max_length = max(sub_lengths)

            lengths.append(np.asarray(sub_lengths))
            max_lengths.append(max_length)
            data.append(sub_data)
            labels.append(sub_labels)
        
        self.data = data
        self.lengths = lengths
        self.max_lengths = max_lengths
        self.labels = labels

    def split(self, mode='segment', time_steps=10, subject_ids=[0], train_test_point=9):
        assert mode in ['segment', 'whole']

        train_X = []
        train_y = []
        train_lens = []
        
        test_X = []
        test_y = []
        test_lens = []

        for subject_id in subject_ids:
            train_X.extend(self.data[subject_id][:train_test_point])
            train_y.extend(self.labels[subject_id][:train_test_point]) 
            train_lens.extend(self.lengths[subject_id][:train_test_point])

            test_X.extend(self.data[subject_id][train_test_point:])
            test_y.extend(self.labels[subject_id][train_test_point:])
            test_lens.extend(self.lengths[subject_id][train_test_point:])
        
        max_length = max(self.max_lengths)
        assert len(train_X) == 9 * len(subject_ids)
        train_batches = None
        test_batches = None

        if mode == 'segment':
            train_X, train_y, train_lens, train_batches  = segment(train_X, train_y, train_lens, time_steps)
            test_X, test_y, test_lens, test_batches = segment(test_X, test_y, test_lens, time_steps)

            print('train_X shape: {}'.format(train_X.shape))
            print('train_y shape: {}'.format(train_y.shape))
            print('train_lens shape: {}'.format(train_lens.shape))

            print('test_X shape: {}'.format(test_X.shape))
            print('test_y shape: {}'.format(test_y.shape))
            print('test_lens shape: {}'.format(test_lens.shape))

        else:
            train_X = pad_seq(train_X, max_length)
            train_X = np.stack(train_X)
            train_y = np.stack(train_y)
            train_lens = np.stack(train_lens)

            test_X = pad_seq(test_X, max_length)
            test_X = np.stack(test_X)
            test_y = np.stack(test_y)
            test_lens = np.stack(test_lens)

            print('train_X shape: {}'.format(train_X.shape))
            print('train_y shape: {}'.format(train_y.shape))
            print('train_lens shape: {}'.format(train_lens.shape))

            print('test_X shape: {}'.format(test_X.shape))
            print('test_y shape: {}'.format(test_y.shape))
            print('test_lens shape: {}'.format(test_lens.shape))
        return train_X, train_y, train_lens, train_batches, test_X, test_y, test_lens, test_batches

    def get_batch(self, sub_data, sub_labels, sub_lengths, batch_size=2):
        batch_data = None
        batch_lengths = None
        batch_labels = None

        if batch_size == -1:
            batch_data = sub_data
            batch_lengths = sub_lengths
            batch_labels = sub_labels
        else:
            n_samples = sub_data.shape[0]
            indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_data = sub_data[indices]
            batch_lengths = sub_lengths[indices]
            batch_labels = sub_labels[indices]
        
        return batch_data, batch_labels, batch_lengths

if __name__ == '__main__':
    dataset = EGGDataset()
    dataset.read_data()
    train_X, train_y, train_lens, train_batches, test_X, test_y, test_lens, test_batches = dataset.split(mode='segment', time_steps=100, subject_ids=[0,1,2])  
