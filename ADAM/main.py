import class_model.dataset_mode as dm
import class_model.mlp_model as mm
import ADAM.adam_model as am
import ADAM.dataset_office31 as do


# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())

def main():
    # od = do.Office31Dataset()
    data = dm.Select('flower', 'select')
    # om1 = mm.MlpModel("office31_model_1", data, [10])
    # om1.exec_all(epoch_count=20, report=10)
    om2 = am.AdamModel('office31_model_2',data,[64,32,10])
    om2.exec_all(epoch_count=50,report=10,learning_rate=0.0001)
    #

if __name__ == '__main__':
    # randomize()
    main()
