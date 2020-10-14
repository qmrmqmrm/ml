import numpy as np
import csv
from ABALONE.CODE.abalone import Abalone
from PULSAR.CODE.pulsar import Pulsar
from STEEL.CODE.steel import Steel

np.random.seed(1234)


# def randomize(): np.random.seed(time.time())

class Mlp(Pulsar):
    def __init__(self):

        self.RND_MEAN = 0
        self.RND_STD = 0.0030
        self.LEARNING_RATE = 0.001
        self.input_cnt = None
        self.output_cnt = None
        self.hidden_cnt = None
        self.pm_hidden = None
        self.pm_output = None
        self.hidden_config = None

    def init_model_hidden1(self):
        # print("mlp:init_model_hidden1")
        # print(self.input_cnt)
        self.pm_hidden = self.alloc_param_pair([self.input_cnt, self.hidden_cnt])
        print("은닉층 weight shape",self.pm_hidden["w"].shape)
        self.pm_output = self.alloc_param_pair([self.hidden_cnt, self.output_cnt])
        print("출력층 weight shape",self.pm_output["w"].shape)

    def alloc_param_pair(self, shape):
        # print("mlp:alloc_param_pair")
        weight = np.random.normal(self.RND_MEAN, self.RND_STD, shape)
        # print(weight.shape)
        bias = np.zeros(shape[-1])

        return {'w': weight, 'b': bias}

    def forward_neuralnet_hidden1(self, x):
        # print("mlp:forward_neuralnet_hidden1")
        hidden = self.relu(np.matmul(x, self.pm_hidden['w']) + self.pm_hidden['b'])
        # print("hidden:",hidden.shape)
        output = np.matmul(hidden, self.pm_output['w']) + self.pm_output['b']
        # print(output.shape)
        return output, [x, hidden]

    def relu(self, x):
        # print("mlp:relu")
        return np.maximum(x, 0)

    def backprop_neuralnet_hidden1(self, G_output, aux):
        # print("mlp:backprop_neuralnet_hidden1")
        x, hidden = aux
        # print(hidden.shape,G_output.shape)
        g_output_w_out = hidden.transpose()

        # print(g_output_w_out.shape)
        G_w_out = np.matmul(g_output_w_out, G_output)
        G_b_out = np.sum(G_output, axis=0)
        # print(G_w_out.shape)
        g_output_hidden = self.pm_output['w'].transpose()
        G_hidden = np.matmul(G_output, g_output_hidden)
        # print((G_hidden.shape))
        self.pm_output['w'] -= self.LEARNING_RATE * G_w_out
        self.pm_output['b'] -= self.LEARNING_RATE * G_b_out

        G_hidden = G_hidden * self.relu_derv(hidden)

        g_hidden_w_hid = x.transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)

        self.pm_hidden['w'] -= self.LEARNING_RATE * G_w_hid
        self.pm_hidden['b'] -= self.LEARNING_RATE * G_b_hid

    def relu_derv(self, y):
        # print("mlp:relu_derv")
        return np.sign(y)

    def init_model_hiddens(self):
        # print("mlp:init_model_hiddens")
        self.pm_hiddens = []
        prev_cnt = self.input_cnt
        # print("input_cnt",self.input_cnt)
        # print(prev_cnt)

        for i, hidden_cnt in enumerate(self.hidden_config):

            # print("hidden_cnt",hidden_cnt)
            self.pm_hiddens.append(self.alloc_param_pair([prev_cnt, hidden_cnt]))
            print(f"은닉층{i+1} weight shape{self.pm_hiddens[i]['w'].shape}")
            prev_cnt = hidden_cnt
            # print("prev_cnt,",prev_cnt)
        self.pm_output = self.alloc_param_pair([prev_cnt, self.output_cnt])
        print("출력층 weight shape", self.pm_output["w"].shape)

    def forward_neuralnet_hiddens(self, x):
        # print("mlp:forward_neuralnet_hiddens")
        hidden = x
        hiddens = [x]

        for pm_hidden in self.pm_hiddens:
            # print("은닉층 weight shape",pm_hidden['w'].shape)
            hidden = self.relu(np.matmul(hidden, pm_hidden['w']) + pm_hidden['b'])
            # print("hidden:",hidden.shape)
            hiddens.append(hidden)
        # print("fin for")
        output = np.matmul(hidden, self.pm_output['w']) + self.pm_output['b']


        return output, hiddens

    def backprop_neuralnet_hiddens(self, G_output, aux):
        # print("mlp:backprop_neuralnet_hiddens")
        hiddens = aux
        # print(range(len(self.pm_hiddens)))
        g_output_w_out = hiddens[-1].transpose()
        G_w_out = np.matmul(g_output_w_out, G_output)
        G_b_out = np.sum(G_output, axis=0)

        g_output_hidden = self.pm_output['w'].transpose()
        G_hidden = np.matmul(G_output, g_output_hidden)

        self.pm_output['w'] -= self.LEARNING_RATE * G_w_out
        self.pm_output['b'] -= self.LEARNING_RATE * G_b_out

        for n in reversed(range(len(self.pm_hiddens))):
            # print(n)
            # print(len(hiddens))
            G_hidden = G_hidden * self.relu_derv(hiddens[n + 1])


            g_hidden_w_hid = hiddens[n].transpose()
            G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
            G_b_hid = np.sum(G_hidden, axis=0)

            g_hidden_hidden = self.pm_hiddens[n]['w'].transpose()
            G_hidden = np.matmul(G_hidden, g_hidden_hidden)

            self.pm_hiddens[n]['w'] -= self.LEARNING_RATE * G_w_hid
            self.pm_hiddens[n]['b'] -= self.LEARNING_RATE * G_b_hid
        # print("fin for")

    def init_model(self):
        print("mlp:init_model")
        # print(self.hidden_config)
        if self.hidden_config is not None:
            print('은닉 계층 {}개를 갖는 다층 퍼셉트론이 작동되었습니다.'. \
                  format(len(self.hidden_config)))
            self.init_model_hiddens()
        else:
            print('은닉 계층 하나를 갖는 다층 퍼셉트론이 작동되었습니다.')
            self.init_model_hidden1()

    def forward_neuralnet(self, x):
        # print("mlp:forward_neuralnet")
        # print(x.shape)
        if self.hidden_config is not None:
            return self.forward_neuralnet_hiddens(x)
        else:
            return self.forward_neuralnet_hidden1(x)

    def backprop_neuralnet(self, G_output, hiddens):
        # print("mlp:backprop_neuralnet")
        if self.hidden_config is not None:
            self.backprop_neuralnet_hiddens(G_output, hiddens)
        else:
            self.backprop_neuralnet_hidden1(G_output, hiddens)

    def set_hidden(self, info):
        # print("mlp:set_hidden")
        if isinstance(info, int):
            # print(f"if:{info}")
            self.hidden_cnt = info
            self.hidden_config = None
        else:
            self.hidden_config = info

    # def abalone_exec(self, epoch_count=10, mb_size=20, report=1):
    #     self.load_abalone_dataset()
    #     self.init_model()
    #     self.train_and_test(epoch_count, mb_size, report)

    # def pulsar_exec(self, epoch_count=10, mb_size=334, report=1):
    #     self.load_pulsar_dataset()
    #     self.init_model()
    #     self.train_and_test(epoch_count, mb_size, report)
    #
    # def steel_exec(self, epoch_count=10, mb_size=10, report=1):
    #     self.load_steel_dataset()
    #     self.init_model()
    #     self.train_and_test(epoch_count, mb_size, report)

def main():
    mlp = Mlp()
    mlp.set_hidden([12,6])
    # mlp.abalone_exec(epoch_count=50 ,report=10)
    # mlp.abalone_exec(epoch_count=50, report=10)
    mlp.pulsar_exec(epoch_count=50,report=10)
    # mlp.steel_exec(epoch_count=50,report=10)


if __name__ == '__main__':
    main()
