### Documentation to Setup Standalone Distributed XGBoost 
1. Clone the dmlc-core library <br> <br>
`git clone https://github.com/dmlc/dmlc-core.git`

2. Install the relevant Python packages. <br> <br>
`pip3 install xgboost` <br>
`pip3 install kubernetes` <br> <br>
Note that we install the Kubernetes Python package so that the `dmlc-submit` script doesn't error out when we run it. `dmlc-submit` imports Kubernetes, but we won't be using it.

3. Clone this repo <br> <br>
`git clone https://github.com/chester-leung/mc2.git`

4. The `mc2/xgb_standalone/tutorial` folder contains the training and test script that we'll be using for this tutorial, along with the training and test data. Note that if passing a path to the data to `xgb.DMatrix`, the data must be in LibSVM form. Now simulate the distributed training locally with 3 workers, each with 3 GB memory. <br> <br>
`dmlc-core/tracker/dmlc-submit --cluster local --num-workers 3 --worker-memory 3g python3 xgb_standalone/tutorial/standalone_xgb_sample.py`

5. The training and evaluation should finish in under 10 seconds. I obtained a Root Mean Squared Error of `4.624654`.


