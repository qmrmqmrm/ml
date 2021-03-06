from class_model.mlp_model import MlpModel
import class_model.mathutil as mu
import numpy as np


class AdamModel(MlpModel):
    def __init__(self, name, dataset, hconfigs,use_adam =True):
        # print("adam model init")
        self.use_adam = use_adam
        super(AdamModel, self).__init__(name, dataset, hconfigs)

    def backprop_layer(self, G_y, hconfig, pm, aux):
        # print("adam backprop_layer")
        x, y = aux
        if hconfig is not None: G_y = mu.relu_derv(y) * G_y

        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()

        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)

        self.update_param(pm, 'w', G_weight)  # pm['w'] -= self.learning_rate * G_weight
        self.update_param(pm, 'b', G_bias)  # pm['b'] -= self.learning_rate * G_bias

        return G_input

    def update_param(self, pm, key, delta):
        # print("adam update_param")
        if self.use_adam:                   # True 이면 아담 업데이트 시작
            delta = self.eval_adam_delta(pm, key, delta)
        pm[key] -= self.learning_rate * delta

    def eval_adam_delta(self, pm, key, delta):
        ro_1 = 0.9
        ro_2 = 0.999
        epsilon = 1.0e-8

        skey, tkey, step = 's' + key, 't' + key, 'n' + key
        if skey not in pm:
            pm[skey] = np.zeros(pm[key].shape)
            pm[tkey] = np.zeros(pm[key].shape)
            pm[step] = 0

        s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
        t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)

        pm[step] += 1
        s = s / (1 - np.power(ro_1, pm[step]))
        t = t / (1 - np.power(ro_2, pm[step]))

        return s / (np.sqrt(t) + epsilon)



