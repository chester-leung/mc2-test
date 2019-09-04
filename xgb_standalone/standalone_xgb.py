import xgboost as xgb

def main():
    dtrain = xgb.DMatrix('/home/ubuntu/mc2/msd_training_data_libsvm.data')
    dtest = xgb.DMatrix('/home/ubuntu/mc2/msd_test_data_libsvm.data')

    params = {'max_depth': 3, 'min_child_weight': 1.0, 'lambda': 1.0}
    num_rounds = 100
    
    model = xgb.train(params, dtrain, num_rounds)
    predictions = model.predict(dtest)
    print(model.eval(dtest))

    if xgb.rabit.get_rank() == 0:
        model.save_model('saved.model')

if __name__ == '__main__':
    xgb.rabit.init()

    n_workers = xgb.rabit.get_world_size()
    rank = xgb.rabit.get_rank()
    
    main()

    xgb.rabit.finalize()
