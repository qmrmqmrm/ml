import numpy as np
import class_model.dataset_mode as dm
import class_model.mlp_model as mm
import time

# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())


def main():
    data = dm.Select('steel', 'select')
    model = mm.MlpModel("abalone_model", data, [12,4,6])
    model.exec_all(epoch_count=50, report=10, learning_rate=0.0001)


if __name__ == '__main__':
    # randomize()
    main()


