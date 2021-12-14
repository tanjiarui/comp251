import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_svmlight_file
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()
# load data
feature, label = load_svmlight_file('sample libsvm data.txt')
feature = pd.DataFrame(feature.todense())
label = pd.DataFrame(label, columns=['label'])
df_terry = spark.createDataFrame(pd.concat([feature, label], axis=1), ['feature ' + str(i) for i in range(feature.shape[1])] + ['label'])  # switch pandas to spark
print('row number: %d' % df_terry.count())
print('column number: %d' % len(df_terry.columns))
df_terry.show(4)

# preprocessing
df_terry = RFormula(formula='label ~ .').fit(df_terry).transform(df_terry).select(['features', 'label'])
train_terry, test_terry = df_terry.randomSplit([.65, .35])
label_indexer = StringIndexer(inputCol='label', outputCol='indexed_label_terry', stringOrderType='alphabetAsc').fit(df_terry)
feature_indexer = VectorIndexer(maxCategories=4, inputCol='features', outputCol='indexed_features_terry').fit(df_terry)
print('input column: ' + feature_indexer.getInputCol())
print('output column: ' + feature_indexer.getOutputCol())
print('map of categories:')
print(feature_indexer.categoryMaps)

# modeling
dt_terry = DecisionTreeClassifier(featuresCol='indexed_features_terry', labelCol='indexed_label_terry')
pipeline_terry = Pipeline(stages=[label_indexer, feature_indexer, dt_terry])
model_terry = pipeline_terry.fit(train_terry)
prediction_terry = model_terry.transform(test_terry)
prediction_terry.printSchema()
prediction_terry.show(10)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='accuracy')
print('accuracy: %.2f' % evaluator.evaluate(prediction_terry))