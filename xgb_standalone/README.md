Experiments exploring whether manually sending data to different workers actually helps

Using entirety of dataset, manually sending different third of data to each node. <br>
Rank 0 got 154571 rows, Rank 1 got 154571 rows, Rank 2 got 154572 rows <br>
RMSE = 9.179495 <br>
*Note that this is the most similar to the federated setting* <br>
<br>
Using entirety of dataset, calling DMatrix to split data across nodes. <br>
Rank 0 got 154578 rows, Rank 1 got 154570 rows, Rank 2 got 154567 rows <br>
RMSE = 9.183995 <br>
<br>
Using one third of dataset, manually sending same third of data to each node. <br>
Rank 0, Rank 1, Rank 2 all got 154571 rows <br>
RMSE = 9.875301 <br>
<br>
Using one third of dataset, calling DMatrix to split data across nodes. <br>
Rank 0 got 51519 rows, Rank 1 got 51520 rows, Rank 2 got 51533 rows <br>
RMSE = 9.220065 <br>
