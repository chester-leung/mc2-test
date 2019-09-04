import org.apache.spark.sql.SparkSession 
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions.rand


val spark = SparkSession.builder().getOrCreate()

// Train the Model
val schema = new StructType(Array(
    StructField("year", DoubleType, true),
    StructField("1", DoubleType, true),
	StructField("2", DoubleType, true),
	StructField("3", DoubleType, true),
	StructField("4", DoubleType, true),
	StructField("5", DoubleType, true),
	StructField("6", DoubleType, true),
	StructField("7", DoubleType, true),
	StructField("8", DoubleType, true),
	StructField("9", DoubleType, true),
	StructField("10", DoubleType, true),
	StructField("11", DoubleType, true),
	StructField("12", DoubleType, true),
	StructField("13", DoubleType, true),
	StructField("14", DoubleType, true),
	StructField("15", DoubleType, true),
	StructField("16", DoubleType, true),
	StructField("17", DoubleType, true),
	StructField("18", DoubleType, true),
	StructField("19", DoubleType, true),
	StructField("20", DoubleType, true),
	StructField("21", DoubleType, true),
	StructField("22", DoubleType, true),
	StructField("23", DoubleType, true),
	StructField("24", DoubleType, true),
	StructField("25", DoubleType, true),
	StructField("26", DoubleType, true),
	StructField("27", DoubleType, true),
	StructField("28", DoubleType, true),
	StructField("29", DoubleType, true),
	StructField("30", DoubleType, true),
	StructField("31", DoubleType, true),
	StructField("32", DoubleType, true),
	StructField("33", DoubleType, true),
	StructField("34", DoubleType, true),
	StructField("35", DoubleType, true),
	StructField("36", DoubleType, true),
	StructField("37", DoubleType, true),
	StructField("38", DoubleType, true),
	StructField("39", DoubleType, true),
	StructField("40", DoubleType, true),
	StructField("41", DoubleType, true),
	StructField("42", DoubleType, true),
	StructField("43", DoubleType, true),
	StructField("44", DoubleType, true),
	StructField("45", DoubleType, true),
	StructField("46", DoubleType, true),
	StructField("47", DoubleType, true),
	StructField("48", DoubleType, true),
	StructField("49", DoubleType, true),
	StructField("50", DoubleType, true),
	StructField("51", DoubleType, true),
	StructField("52", DoubleType, true),
	StructField("53", DoubleType, true),
	StructField("54", DoubleType, true),
	StructField("55", DoubleType, true),
	StructField("56", DoubleType, true),
	StructField("57", DoubleType, true),
	StructField("58", DoubleType, true),
	StructField("59", DoubleType, true),
	StructField("60", DoubleType, true),
	StructField("61", DoubleType, true),
	StructField("62", DoubleType, true),
	StructField("63", DoubleType, true),
	StructField("64", DoubleType, true),
	StructField("65", DoubleType, true),
	StructField("66", DoubleType, true),
	StructField("67", DoubleType, true),
	StructField("68", DoubleType, true),
	StructField("69", DoubleType, true),
	StructField("70", DoubleType, true),
	StructField("71", DoubleType, true),
	StructField("72", DoubleType, true),
	StructField("73", DoubleType, true),
	StructField("74", DoubleType, true),
	StructField("75", DoubleType, true),
	StructField("76", DoubleType, true),
	StructField("77", DoubleType, true),
	StructField("78", DoubleType, true),
	StructField("79", DoubleType, true),
	StructField("80", DoubleType, true),
	StructField("81", DoubleType, true),
	StructField("82", DoubleType, true),
	StructField("83", DoubleType, true),
	StructField("84", DoubleType, true),
	StructField("85", DoubleType, true),
	StructField("86", DoubleType, true),
	StructField("87", DoubleType, true),
	StructField("88", DoubleType, true),
	StructField("89", DoubleType, true),
	StructField("90", DoubleType, true)))

val training_data_df = spark.read.schema(schema).csv("/home/ubuntu/mc2/msd_training_data.csv")

val vectorAssembler = new VectorAssembler().
  setInputCols(Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90"
)).
  setOutputCol("features")

val transformed_df = vectorAssembler.transform(training_data_df)

// Select two columns and shuffle data
val xgbInput = transformed_df.select("features", "year").orderBy(rand())

val xgbParam = Map(
	"eta" -> 0.1,
	"max_depth" -> 3,
	"num_round" -> 100,
	"num_workers" -> 2
)

val xgbRegressor = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol("year")
val xgbModel = xgbRegressor.fit(xgbInput)

// Evaluation
val test_data_df = spark.read.schema(schema).csv("/home/ubuntu/mc2/msd_test_data.csv")
val test_set = vectorAssembler.transform(test_data_df).select("features", "year")
val predictions = xgbModel.transform(test_set)

// Root Mean Absolute Error
val mae_evaluator = new RegressionEvaluator().setMetricName("mae").setLabelCol("year").setPredictionCol("prediction")

val mae = mae_evaluator.evaluate(predictions)
println(s"Mean Absolute Error: $mae")

// Mean Squared Error
val rmse_evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("year").setPredictionCol("prediction")

val rmse = rmse_evaluator.evaluate(predictions)
println(s"Root Mean Squared Error: $rmse")


// Train Second Model with Biased Data
println("\n\n------------ Training with Bias --------------\n\n")
val biased_transformed_df = transformed_df.sort($"year")
val biased_xgbInput = biased_transformed_df.select("features", "year")

println("Unbiased Input")
xgbInput.show()
println("Biased Input")
biased_xgbInput.show()

val biased_xgbRegressor = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol("year")
val biased_xgbModel = biased_xgbRegressor.fit(biased_xgbInput)

// Evaluation
val biased_predictions = biased_xgbModel.transform(test_set)

// Mean Absolute Error
val biased_mae_evaluator = new RegressionEvaluator().setMetricName("mae").setLabelCol("year").setPredictionCol("prediction")

val biased_mae = biased_mae_evaluator.evaluate(biased_predictions)
println(s"Mean Absolute Error with Biased Dataset: $biased_mae")

// Root Mean Squared Error
val biased_rmse_evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("year").setPredictionCol("prediction")

val biased_rmse = biased_rmse_evaluator.evaluate(biased_predictions)
println(s"Root Mean Squared Error: $biased_rmse")
