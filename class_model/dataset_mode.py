from class_model.dataset import DataSet
import numpy as np
import class_model.mathutil as mu


class Regression(DataSet):
    def __init__(self, name):
        super(Regression, self).__init__(name)

    def dataset_forward_postproc(self, output, y):
        # print("regression dataset_forward_postproc")
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        aux = diff

        return loss, aux

    def dataset_backprop_postproc(self, G_loss, aux):
        # print("regression dataset_backprop_post_proc")
        diff = aux
        shape = diff.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff

        return G_output

    def dataset_eval_accuracy(self, x, y, output):
        # print("regression dataset_eval_accuracy")
        mse = np.mean(np.square(output - y))
        accuracy = 1 - np.sqrt(mse) / np.mean(y)

        return accuracy

    def dataset_get_estimate(self, output):
        # print("regression dataset_get_estimate")
        estimate = output

        return estimate


class Binary(DataSet):
    def __init__(self, name):
        super(Binary, self).__init__(name)

    def dataset_forward_postproc(self, output, y):
        # print("Binary dataset_forward_postproc")
        entropy = mu.sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [y, output]

        return loss, aux

    def dataset_backprop_postproc(self, G_loss, aux):
        # print("Binary dataset_backprop_post_proc")
        y, output = aux
        shape = output.shape

        g_loss_entropy = np.ones(shape) / np.prod(shape)
        g_entropy_output = mu.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def dataset_eval_accuracy(self, x, y, output):
        # print("Binary dataset_eval_accuracy")
        estimate = np.greater(output, 0)
        answer = np.equal(y, 1.0)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

        return accuracy

    def dataset_get_estimate(self, output):
        # print("Binary dataset_get_estimate")

        estimate = mu.sigmoid(output)

        return estimate


class Select(DataSet):
    def __init__(self, name):
        super(Select, self).__init__(name)

    def dataset_forward_postproc(self, output, y):
        # print("Select dataset_forward_postproc")
        losses = list()
        auxes = list()

        if self.name == "office31":
            # print("y len : ",len(y))
            # print("output len : ", len(output))
            output, y = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

            for i in range(2):
                entropy = mu.softmax_cross_entropy_with_logits(y[i], output[i])
                loss = np.mean(entropy)
                losses.append(loss)
                aux = [output[i], y[i], entropy]
                auxes.append(aux)
            # loss0 + loss1, [aux0, aux1]
            total_loss = losses[0] + losses[1]
            return total_loss, auxes

        entropy = mu.softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [output, y, entropy]

        return loss, aux


    def dataset_backprop_postproc(self, G_loss, aux):
        # print("Select dataset_backprop_postproc")
        G_outputs = list()
        if self.name == "office31":
            for i in range(2):
                output, y, entropy = aux[i]
                g_loss_entropy = 1.0 / np.prod(entropy.shape)
                g_entropy_output = mu.softmax_cross_entropy_with_logits_derv(y, output)

                G_entropy = g_loss_entropy * G_loss
                G_output = g_entropy_output * G_entropy
                G_outputs.append(G_output)


            return np.hstack([G_outputs[0], G_outputs[1]])


        # print("Select dataset_backprop_postproc")
        output, y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = mu.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def dataset_eval_accuracy(self, x, y, output):
        # print("Select dataset_eval_accuracy")
        accuracies = list()
        if self.name == "office31":
            output, y = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)
            # print("eval len : ", np.array(output).shape)
            for i in range(2):
                # print("Select dataset_eval_accuracy")
                estimate = np.argmax(output[i], axis=1)  # 각 class의 최대값(10)
                answer = np.argmax(y[i], axis=1)  # 정답값 최대값(10)
                correct = np.equal(estimate, answer)  # estimate가 같으면 true, 다르면 false
                accuracy = np.mean(correct)
                accuracies.append(accuracy)
            return accuracies


        # print("Select dataset_eval_accuracy")
        estimate = np.argmax(output, axis=1)  # 각 class의 최대값(10)
        answer = np.argmax(y, axis=1)  # 정답값 최대값(10)
        correct = np.equal(estimate, answer)  # estimate가 같으면 true, 다르면 false
        accuracy = np.mean(correct)

        return accuracy

    def dataset_get_estimate(self, output):
        # print("Select dataset_get_estimate")
        estimates = list()
        if self.name == "office31":
            output = np.hsplit(output, self.cnts)
            for i in range(2):
                estimate = mu.softmax(output[i])
                estimates.append(estimate)

            return np.hstack([estimates[0], estimates[1]])

        estimate = mu.softmax(output)

        return estimate