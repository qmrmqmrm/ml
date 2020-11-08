import numpy as np
import class_model.dataset_mode as dm
import class_model.mlp_model as mm

np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())


def main():
    data = dm.Select('flower', 'select')
    model = mm.MlpModel("flower_model", data, [10])
    model.exec_all(epoch_count=50, report=10)


if __name__ == '__main__':
    main()


