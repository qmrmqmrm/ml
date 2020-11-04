import numpy as np


class DatasetBase:
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name, self.mode, len(self.tr_xs), len(self.te_xs), len(self.va_xs))

    def dataset_get_train_data(self, batch_size, nth):
        pass

    def dataset_get_test_data(self):
        pass

    def dataset_shuffle_train_data(self, size):
        pass

    def dataset_get_validate_data(self, count):
        pass

    def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        pass

    def visualize(self, xs, estimates, answers):
        pass

    def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs),
                                                                                         np.mean(accs), acc, time1,
                                                                                         time2))

    def dataset_test_prt_result(self, name, acc, time):
        print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'.format(name, acc, time))

    @property
    def train_count(self):
        return len(self.tr_xs)

