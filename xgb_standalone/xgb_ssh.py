import xgboost as xgb
from numpy import genfromtxt

def main():
    data = genfromtxt('/home/ubuntu/mc2/data/msd_training_data_split_sample.csv', delimiter=',')
    #labels = data[:, 0]
    #data = data[:, 1:]
    dtrain = xgb.DMatrix(data[:, 1:], label=data[:, 0])

    print("Rank: {}, dtrain rows: {}".format(xgb.rabit.get_rank(), dtrain.num_row()))
    params = {'max_depth': 3, 'min_child_weight': 1.0, 'lambda': 1.0}
    num_rounds = 20
    
    model = xgb.train(params, dtrain, num_rounds)

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
    xgb.rabit.init()

    n_workers = xgb.rabit.get_world_size()
    rank = xgb.rabit.get_rank()
    
    main()

    xgb.rabit.finalize()

    
