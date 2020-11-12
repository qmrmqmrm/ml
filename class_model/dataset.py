from class_model.dataset_base import DatasetBase
import class_model.mathutil as mu
import numpy as np
import os


class DataSet(DatasetBase):
    def __init__(self, name, mode):
        super(DataSet, self).__init__(name, mode)
        # print("dataset init")
        resolution = [100, 100]
        input_shape = [-1]
        if self.name == 'abalone':
            rows, _ = mu.load_csv('../data/abalone.csv')

            xs = np.zeros([len(rows), 10])
            ys = np.zeros([len(rows), 1])

            for n, row in enumerate(rows):
                if row[0] == 'I':
                    xs[n, 0] = 1
                if row[0] == 'M':
                    xs[n, 1] = 1
                if row[0] == 'F':
                    xs[n, 2] = 1
                xs[n, 3:] = row[1:-1]
                ys[n, :] = row[-1:]

            self.dataset_shuffle_data(xs, ys, 0.8)

        elif self.name == 'pulsar':
            rows, _ = mu.load_csv('../data/pulsar_stars.csv')
            data = np.asarray(rows, dtype='float32')
            self.dataset_shuffle_data(data[:, :-1], data[:, -1:], 0.8)
            self.target_names = ['별', '펄서']

        elif self.name == 'steel':
            rows, headers = mu.load_csv('../data/faults.csv')
            data = np.asarray(rows, dtype='float32')
            self.dataset_shuffle_data(data[:, :-7], data[:, -7:], 0.8)

            self.target_names = headers[-7:]

        elif self.name == 'pulsarselect':
            rows, _ = mu.load_csv('../data/pulsar_stars.csv')
            data = np.asarray(rows, dtype='float32')
            self.dataset_shuffle_data(data[:, :-1], mu.onehot(data[:, -1], 2), 0.8)
            self.target_names = ['별', '펄서']

        elif self.name == 'flower':

            path = '../data/flowers'
            self.target_names = mu.list_dir(path)

            images = []
            idxs = []

            for dx, dname in enumerate(self.target_names):
                subpath = path + '/' + dname
                filenames = mu.list_dir(subpath)
                for fname in filenames:
                    if fname[-4:] != '.jpg':
                        continue
                    imagepath = os.path.join(subpath, fname)
                    pixels = mu.load_image_pixels(imagepath, resolution, input_shape)
                    images.append(pixels)
                    idxs.append(dx)
            self.image_shape = resolution + [3]
            xs = np.asarray(images, np.float32)
            ys = mu.onehot(idxs, len(self.target_names))
            self.dataset_shuffle_data(xs, ys, 0.8)

        elif self.name == 'office31':
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
            print(type(self.target_names[0]))
            self.cnts = [len(domain_names)]

    def dataset_get_train_data(self, batch_size, nth):
        # print("dataset dataset_get_train_data")
        from_idx = nth * batch_size
        to_idx = (nth + 1) * batch_size

        tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
        # print(tr_X)
        tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]
        # print(tr_X.shape, tr_Y.shape)
        return tr_X, tr_Y

    def dataset_get_test_data(self):
        # print("dataset dataset_get_test_data")
        return self.te_xs, self.te_ys

    def dataset_shuffle_train_data(self, size):
        # print("dataset dataset_shuffle_train_data")
        self.indices = np.arange(size)
        # print(self.indices)
        np.random.shuffle(self.indices)

    def dataset_get_validate_data(self, count):
        # print("dataset dataset_get_validate_data")
        self.va_indices = np.arange(len(self.va_xs))  # (216)
        np.random.shuffle(self.va_indices)

        va_X = self.va_xs[self.va_indices[0:count]]  # (3,30000)
        va_Y = self.va_ys[self.va_indices[0:count]]  # (3,5)

        return va_X, va_Y

    def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
        # print("dataset dataset_shuffle_data")
        data_count = len(xs)  # 4323

        tr_cnt = int(data_count * tr_ratio / 10) * 10  # 3450
        va_cnt = int(data_count * va_ratio)  # 216
        # print(data_count)
        te_cnt = data_count - (tr_cnt + va_cnt)

        tr_from, tr_to = 0, tr_cnt
        va_from, va_to = tr_cnt, tr_cnt + va_cnt
        te_from, te_to = tr_cnt + va_cnt, data_count

        indices = np.arange(data_count)  # 4323개의 array생성
        np.random.shuffle(indices)

        self.tr_xs = xs[indices[tr_from:tr_to]]  # (3450, 30000)
        self.tr_ys = ys[indices[tr_from:tr_to]]  # (3450, 5)
        self.va_xs = xs[indices[va_from:va_to]]
        self.va_ys = ys[indices[va_from:va_to]]
        self.te_xs = xs[indices[te_from:te_to]]  # (657,30000)
        self.te_ys = ys[indices[te_from:te_to]]  # (657,5)

        self.input_shape = xs[0].shape  # 30000
        # print(self.input_shape)
        self.output_shape = ys[0].shape  # 5
        return indices[tr_from:tr_to], indices[va_from:va_to], indices[te_from:te_to]

    def dataset_forward_postproc(self, output, y):
        pass

    def dataset_backprop_postproc(self, G_loss, aux):
        pass

    def dataset_eval_accuracy(self, x, y, output):
        pass

    def dataset_get_estimate(self, output):
        pass

    def visualize(self, xs, estimates, answers):
        # print("dataset visualize")
        # print(f"self.name{self.name}")
        if self.name == 'abalone':
            for n in range(len(xs)):
                x, est, ans = xs[n], estimates[n], answers[n]
                xstr = mu.vector_to_str(x, '%4.2f')
                print('{} => 추정 {:4.1f} : 정답 {:4.1f}'.
                      format(xstr, est[0], ans[0]))
        elif self.name == 'pulsar':

            for n in range(len(xs)):
                x, est, ans = xs[n], estimates[n], answers[n]
                xstr = mu.vector_to_str(x, '%5.1f', 3)
                estr = self.target_names[int(round(est[0]))]
                astr = self.target_names[int(round(ans[0]))]
                rstr = 'O'
                if estr != astr: rstr = 'X'
                print('{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}'.format(xstr, estr, est[0], astr, rstr))

        elif self.name == 'steel':

            mu.show_select_results(estimates, answers, self.target_names)

        elif self.name == 'pulsarselect':

            mu.show_select_results(estimates, answers, self.target_names)

        elif self.name == 'flower':

            mu.draw_images_horz(xs, self.image_shape)
            mu.show_select_results(estimates, answers, self.target_names)


        elif self.name == 'office31':
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