import class_model.dataset_mode as dm
import ADAM.adam_model as am


# np.random.seed(1234)

# def randomize():
#     np.random.seed(time.time())

def main():

    data = dm.Select('office31')
    om2 = am.AdamModel('office31',data,[64,32,10], use_adam=True)
    om2.exec_all(epoch_count=50,report=10,learning_rate=0.0001)



if __name__ == '__main__':
    # randomize()
    main()
