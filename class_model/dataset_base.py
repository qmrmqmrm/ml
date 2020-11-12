import numpy as np


class DatasetBase:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name, self.mode, len(self.tr_xs), len(self.te_xs), len(self.va_xs))

    def dataset_get_train_data(self, batch_size, nth):
        pass

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
        if self.name == 'office31':
            acc_pair = np.mean(accs, axis=0)
            print(f'    Epoch {epoch}: cost={np.mean(costs):5.3f}, accuracy={acc_pair[0]:5.3f}'
                  f'+{acc_pair[1]:5.3f}/{acc[0]:5.3f}+{acc[1]:5.3f} ({time1}/{time2} secs)')

        else:
            print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs),
                                                                                         np.mean(accs), acc, time1,
                                                                                         time2))

    def dataset_test_prt_result(self, name, acc, time):
        if self.name == 'office31':
            print('Model {} test report: accuracy = {:5.3f}+{:5.3f}, ({} secs)\n'.format(name, acc[0], acc[1], time))

        else:
            print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'.format(name, acc, time))

    @property
    def train_count(self):
        return len(self.tr_xs)
