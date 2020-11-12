from class_model.dataset import DataSet
import class_model.mathutil as mu
import class_model.dataset_mode as dm
import os
import numpy as np
from class_model.dataset_mode import Select


# noinspection PyCallByClass
class Office31Dataset(DataSet):
    def __init__(self, resolution=[100, 100], input_shape=[-1]):

        super(Office31Dataset, self).__init__('office31', 'dual_select')

        path = '../data/domain_adaptation_images'
        domain_names = mu.list_dir(path)

        images = []
        didxs, oidxs = [], []
        object_names = None

        for dx, dname in enumerate(domain_names):
            domainpath = os.path.join(path, dname, 'images')
            object_names = mu.list_dir(domainpath)

            for ox, oname in enumerate(object_names):
                objectpath = os.path.join(domainpath, oname)
                filenames = mu.list_dir(objectpath)
                for fname in filenames:
                    if fname[-4:] != '.jpg':
                        continue
                    imagepath = os.path.join(objectpath, fname)
                    pixels = mu.load_image_pixels(imagepath, resolution, input_shape)
                    images.append(pixels)
                    didxs.append(dx)
                    oidxs.append(ox)
        self.image_shape = resolution + [3]

        xs = np.asarray(images, np.float32)  # shape(4110, 30000)

        ys0 = mu.onehot(didxs, len(domain_names))  # ys0.shape(4110, 3)
        ys1 = mu.onehot(oidxs, len(object_names))  # ys1.shape(4110, 31)
        ys = np.hstack([ys0, ys1])  # ys.shape(4110, 34)

        self.dataset_shuffle_data(xs, ys, 0.8)
        self.target_names = [domain_names, object_names]
        self.cnts = [len(domain_names)]

    def dataset_forward_postproc(self, output, y):
        # print("office dataset_forward_postproc")
        outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        loss0, aux0 = Select.dataset_forward_postproc(self, outputs[0], ys[0])
        loss1, aux1 = Select.dataset_forward_postproc(self, outputs[1], ys[1])
        # print(f"loss0{loss0} \n loss1 {loss1}")
        return loss0 + loss1, [aux0, aux1]

    def dataset_backprop_postproc(self, G_loss, aux):
        # print("office dataset_backprop_postproc")
        aux0, aux1 = aux

        G_output0 = Select.dataset_backprop_postproc(self, G_loss, aux0)  # G_output0 (10, 3)
        G_output1 = Select.dataset_backprop_postproc(self, G_loss, aux1)  # G_output1 (10, 31)
        # print(f"G_output {G_output0.shape}, {G_output1.shape}")

        return np.hstack([G_output0, G_output1])

    def dataset_eval_accuracy(self, x, y, output):
        # print("office dataset_eval_accuracy")
        outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)

        acc0 = Select.dataset_eval_accuracy(self, x, ys[0], outputs[0])
        acc1 = Select.dataset_eval_accuracy(self, x, ys[1], outputs[1])

        return [acc0, acc1]

    def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
        # print("office dataset_train_prt_result")
        acc_pair = np.mean(accs, axis=0)
        print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}+{:5.3f}/{:5.3f}+{:5.3f} ({}/{} secs)'.format(epoch,
                                                                                                         np.mean(costs),
                                                                                                         acc_pair[0],
                                                                                                         acc_pair[1],
                                                                                                         acc[0], acc[1],
                                                                                                         time1,
                                                                                                         time2))

    def dataset_test_prt_result(self, name, acc, time):
        # print("office dataset_test_prt_result")

        print('Model {} test report: accuracy = {:5.3f}+{:5.3f}, ({} secs)\n'.format(name, acc[0], acc[1], time))

    def dataset_get_estimate(self, output):
        # print("office get_estimate")
        outputs = np.hsplit(output, self.cnts)

        estimate0 = Select.dataset_get_estimate(self, outputs[0])
        estimate1 = Select.dataset_get_estimate(self, outputs[1])

        return np.hstack([estimate0, estimate1])

    def visualize(self, xs, estimates, answers):

        print(" office visualize ")
        # print(f"estimates{estimates}\n{answers}")
        mu.draw_images_horz(xs, self.image_shape)
        # print(f"estimates type {type(estimates)} shape {estimates.shape}")
        ests, anss = np.hsplit(estimates, self.cnts), np.hsplit(answers, self.cnts)

        captions = ['도메인', '상품']
        # print(f"self.target_names,{len(self.target_names[0])},\n,{len(self.target_names[1])}")
        for m in range(2):
            print('[ {} 추정결과 ]'.format(captions[m]))
            print(f"ests[{m}]{ests[m].shape}")
            mu.show_select_results(ests[m], anss[m], self.target_names[m], 8)
