import numpy as np
import csv
import time

np.random.seed(1234)


# def randomize(): np.random.seed(time.time())

class Abalone():
    def __init__(self):
        self.RND_MEAN = 0
        self.RND_STD = 0.0030
        self.LEARNING_RATE = 0.001
        self.weight = None
        self.bias = None
        self.input_cnt = None
        self.output_cnt = None
        self.data = None
        self.input_cnt = None
        self.output_cnt = None
        self.shuffle_map = None
        self.test_begin_idx = None

    def abalone_exec(self, epoch_count=10, mb_size=10, report=1):
        print("abalone:abalone_exec")
        self.load_abalone_dataset()
        self.init_model()
        self.train_and_test(epoch_count, mb_size, report)

    def load_abalone_dataset(self):
        # print("abalone:load_abalone_dataset")
        with open('C:\\Users\\jm\\Documents\\GitHub\\ml\\ABALONE\\CODE\\abalone.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            rows = []

            for row in csvreader:
                rows.append(row)
            self.input_cnt, self.output_cnt = 10, 1
            self.data = np.zeros([len(rows), self.input_cnt + self.output_cnt])
            for n, row in enumerate(rows):
                if row[0] == 'I': self.data[n, 0] = 1
                if row[0] == 'M': self.data[n, 1] = 1
                if row[0] == 'F': self.data[n, 2] = 1
                self.data[n, 3:] = row[1:]


    def init_model(self):
        # print("abalone:?init_model")
        self.weight = np.random.normal(self.RND_MEAN, self.RND_STD, [self.input_cnt, self.output_cnt])
        self.bias = np.zeros([self.output_cnt])

    def train_and_test(self, epoch_count, mb_size, report):
        # print("abalone:train_and_test")
        step_count = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data()

        for epoch in range(epoch_count):
            losses, accs = [], []

            for n in range(step_count):
                train_x, train_y = self.get_train_data(mb_size, n)
                loss, acc = self.run_train(train_x, train_y)
                losses.append(loss)
                accs.append(acc)

            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                print('Epoch {}: loss={:5.3f}, accuracy = {:5.3f}/{:5.3f}'.
                      format(epoch + 1, np.mean(losses), np.mean(accs), acc))

        final_acc = self.run_test(test_x, test_y)
        print('\nFinal Test: fianl accuracy = {:5.3f}'.format(final_acc))

    def arrange_data(self, mb_size):
        # print("abalone:arrange_data")
        self.shuffle_map = np.arange(self.data.shape[0])
        np.random.shuffle(self.shuffle_map)
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        self.test_begin_idx = step_count * mb_size

        # print(self.test_begin_idx)
        return step_count

    def get_test_data(self):
        # print("abalone:get_test_data")
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
        return test_data[:, :-self.output_cnt], test_data[:, -self.output_cnt:]

    def get_train_data(self, mb_size, nth):
        # print("abalone:get_train_data")
        if nth == 0:
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        train_data = self.data[self.shuffle_map[mb_size * nth:mb_size * (nth + 1)]]

        return train_data[:, :-self.output_cnt], train_data[:, -self.output_cnt:]

    def run_train(self, x, y):
        # print("abalone:run_train")
        output, aux_nn = self.forward_neuralnet(x)
        # print("output :",output.shape)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    def run_test(self, x, y):
        # print("abalone:run_test")
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    def forward_neuralnet(self, x):
        # print("abalone:forward_neuralnet")
        output = np.matmul(x, self.weight) + self.bias
        return output, x

    def backprop_neuralnet(self, G_output, x):
        # print("abalone:backprop_neuralnet")
        g_output_w = x.transpose()

        G_w = np.matmul(g_output_w, G_output)
        G_b = np.sum(G_output, axis=0)
        g = 0
        a = 0
        np.square()
        g += a * g + (1 - a)* G_w ** 2
        self.weight += - self.LEARNING_RATE(G_w / np.sqrt(g) + 1.0e-10)

        self.weight -= self.LEARNING_RATE * G_w
        self.bias -= self.LEARNING_RATE * G_b

    def forward_postproc(self, output, y):
        # print("abalone:forward_postproc")
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        return loss, diff

    def backprop_postproc(self, G_loss, diff):
        # print("abalone:backprop_postproc")
        shape = diff.shape
        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff

        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff
        # print(G_output.shape)
        return G_output

    def backprop_postproc_oneline(self, G_loss, diff):
        # print("abalone:backprop_postproc_oneline")
        return 2 * diff / np.prod(diff.shape)

    def eval_accuracy(self, output, y):
        # print("abalone:eval_accuracy")
        mdiff = np.mean(np.abs((output - y) / y))
        return 1 - mdiff


def main():
    abalone = Abalone()
    abalone.abalone_exec()
    print(f"weight:\n{abalone.weight}")
    print(f"bias:\n{abalone.bias}")


if __name__ == '__main__':
    main()
