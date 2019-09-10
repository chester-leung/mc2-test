import xgboost as xgb
from numpy import genfromtxt

def main():
    dtrain = xgb.DMatrix('/home/ubuntu/mc2/xgb_standalone/tutorial/msd_training_data_sample.libsvm')
    dtest = xgb.DMatrix('/home/ubuntu/mc2/xgb_standalone/tutorial/msd_test_data_sample.libsvm')

    params = {'max_depth': 3, 'min_child_weight': 1.0, 'lambda': 1.0}
    num_rounds = 20
    
    model = xgb.train(params, dtrain, num_rounds)
    predictions = model.predict(dtest)
    print(model.eval(dtest))

if __name__ == '__main__':
    print("init rabit")
    xgb.rabit.init()

    n_workers = xgb.rabit.get_world_size()
    rank = xgb.rabit.get_rank()
    print("start main")
    main()
    print("end main")
    xgb.rabit.finalize()
