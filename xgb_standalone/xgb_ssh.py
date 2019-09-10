print("importing xgb")
import xgboost as xgb
from numpy import genfromtxt

def main():
    print("In start of main")
    data = genfromtxt('/home/ubuntu/mc2/data/msd_training_data_split.csv', delimiter=',')
    print("data genfromtxt")
    #labels = data[:, 0]
    #data = data[:, 1:]
    dtrain = xgb.DMatrix(data[:, 1:], label=data[:, 0])

    print("Rank: {}, dtrain rows: {}".format(xgb.rabit.get_rank(), dtrain.num_row()))
    params = {'max_depth': 3, 'min_child_weight': 1.0, 'lambda': 1.0}
    num_rounds = 20
    
    print("do i make it here")
    model = xgb.train(params, dtrain, num_rounds)
    print("done training")
    if xgb.rabit.get_rank() == 0:
        #model.save_model('saved.model')
        #dtest = xgb.DMatrix('/home/ubuntu/mc2/data/msd_test_data_libsvm.data')
        dtest = xgb.DMatrix('/home/ubuntu/mc2/xgb_standalone/tutorial/msd_test_data_sample.libsvm')

        #test_data = genfromtxt('/home/ubuntu/mc2/data/msd_test_data.csv', delimiter=',')
        #test_labels = test_data[:, 0]
        #test_data = test_data[:, 1:]
        #dtest = xgb.DMatrix(test_data, label=test_labels)
        print(model.eval(dtest))

if __name__ == '__main__':
    print("about to init")
    xgb.rabit.init()
    print("init rabit")

    print("getting world size")
    n_workers = xgb.rabit.get_world_size()
    print("getting rank")
    rank = xgb.rabit.get_rank()
    print("about to main")
    main()
    print("after main")
    xgb.rabit.finalize()

    
