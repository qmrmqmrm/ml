import numpy as np
import csv
import time
# from MLP.CODE.mlp import Mlp
np.random.seed(1234)


# def randomize(): np.random.seed(time.time())

class Steel():
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

    def steel_exec(self, epoch_count=10, mb_size=10, report=1):
        self.load_steel_dataset()
        self.init_model()
        self.train_and_test(epoch_count, mb_size, report)

    def load_steel_dataset(self):
        with open('C:\\Users\\jm\\Documents\\GitHub\\ml\\STEEL\\CODE\\faults.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)


        self.input_cnt, self.output_cnt = 27, 7
        self.data = np.asarray(rows, dtype='float32')


    def init_model(self):
        self.weight = np.random.normal(self.RND_MEAN, self.RND_STD, [self.input_cnt, self.output_cnt])
        self.bias = np.zeros([self.output_cnt])
        # print("b",self.bias.shape)

    def train_and_test(self, epoch_count, mb_size, report):
        step_count = self.arrange_data(mb_size)
        # print(step_count)
        test_x, test_y = self.get_test_data()
        # print(test_x.shape,test_y.shape)

        for epoch in range(epoch_count):
            losses, accs = [], []

            for n in range(step_count):
                # print(f"{n+1} step")
                train_x, train_y = self.get_train_data(mb_size, n)
                # print("train",train_x.shape,train_y.shape)
                loss, acc = self.run_train(train_x, train_y)
                # print(loss,acc)
                losses.append(loss)
                # print(f"len(losses):{len(losses)}")
                accs.append(acc)

            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                print('Epoch {}: loss={:5.3f}, accuracy = {:5.3f}/{:5.3f}'.
                      format(epoch + 1, np.mean(losses), np.mean(accs), acc))

        final_acc = self.run_test(test_x, test_y)
        print('\nFinal Test: fianl accuracy = {:5.3f}'.format(final_acc))

    def arrange_data(self, mb_size):
        self.shuffle_map = np.arange(self.data.shape[0])
        # print(f"{self.shuffle_map.shape}")
        np.random.shuffle(self.shuffle_map)
        # print(int(self.data.shape[0] * 0.8) )
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        # print(f"step{step_count}")
        self.test_begin_idx = step_count * mb_size
        # print(f"self.test_idx{self.test_begin_idx}")
        return step_count

    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
        # print(f"test_data.shape:{test_data.shape}")
        return test_data[:, :-self.output_cnt], test_data[:, -self.output_cnt:]

    def get_train_data(self, mb_size, nth):
        if nth == 0:
            # print(self.shuffle_map[:self.test_begin_idx].shape)
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        # print(mb_size * nth)
        # print(self.shuffle_map[mb_size * nth:mb_size * (nth + 1)])
        train_data = self.data[self.shuffle_map[mb_size * nth:mb_size * (nth + 1)]]

        # print(f"train_data.shape:{train_data.shape}")
        return train_data[:, :-self.output_cnt], train_data[:, -self.output_cnt:]

    def run_train(self, x, y):
        # print(x.shape,y.shape)
        output, aux_nn = self.forward_neuralnet(x)
        # print(f"output{output.shape},aux_nn{aux_nn.shape}")
        loss, aux_pp = self.forward_postproc(output, y)
        # print(f"loss{loss}")
        accuracy = self.eval_accuracy(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        # print(G_output)
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    def forward_neuralnet(self, x):
        # print(x.shape)
        output = np.matmul(x, self.weight) + self.bias
        # print("out",output.shape)
        return output, x

    def backprop_neuralnet(self, G_output, x):
        # print(x.shape)
        g_output_w = x.transpose()

        G_w = np.matmul(g_output_w, G_output)
        G_b = np.sum(G_output, axis=0)

        self.weight -= self.LEARNING_RATE * G_w
        self.bias -= self.LEARNING_RATE * G_b
        # print(self.weight,self.bias)

    def forward_postproc(self, output, y):
        entropy = self.softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        return loss, [y, output, entropy]

    def backprop_postproc(self, G_loss, aux):
        y, output, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output
    def backprop_postproc_oneline(self, G_loss, diff):
        return 2 * diff / np.prod(diff.shape)

    def eval_accuracy(self, output, y):
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(y, axis=1)
        correct = np.equal(estimate, answer)

        return np.mean(correct)
    def softmax(self, x):
        max_elem = np.max(x, axis=1, keepdims=True)
        diff = (x - max_elem)
        exp = np.exp(diff)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        probs = (exp / sum_exp)
        return probs

    def softmax_derv(self,x, y):
        mb_size, nom_size = x.shape
        derv = np.ndarray([mb_size, nom_size, nom_size])
        for n in range(mb_size):
            for i in range(nom_size):
                for j in range(nom_size):
                    derv[n, i, j] = -y[n,i] * y[n,j]
                derv[n, i, i] += y[n,i]
        return derv

    def softmax_cross_entropy_with_logits(self,labels, logits):
        probs = self.softmax(logits)
        return -np.sum(labels * np.log(probs+1.0e-10), axis=1)

    def softmax_cross_entropy_with_logits_derv(self,labels, logits):
        return self.softmax(logits) - labels


def main():
    steel = Steel()
    steel.steel_exec()


if __name__ == '__main__':
    main()
