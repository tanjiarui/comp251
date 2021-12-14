from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, mean, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.getOrCreate()
# load data
wine = spark.read.csv('wine.csv', sep=';', header=True)
for column, dtype in wine.dtypes:
	print('column: %s, type: %s' % (column, dtype))
	print('number of nan')
	wine.select(count(when(isnan(column), column)).alias(column)).show()
wine.summary().show()
wine.select('quality').distinct().show()
wine.groupby('quality').agg(mean('fixed acidity'), mean('volatile acidity'), mean('citric acid'), mean('residual sugar'), mean('chlorides'), mean('free sulfur dioxide'), mean('total sulfur dioxide'), mean('density'), mean('pH'), mean('sulphates'), mean('alcohol')).show()
for column in ['volatile acidity', 'citric acid', 'chlorides', 'sulphates']:
	wine = wine.withColumn(column, col(column).cast('float'))
wine_terry = VectorAssembler(inputCols=['volatile acidity', 'citric acid', 'chlorides', 'sulphates'], outputCol="feature").transform(wine).coalesce(3).cache()

# six clusters
model = KMeans(featuresCol='feature', k=6).fit(wine_terry)
print('cluster sizes')
print(model.summary.clusterSizes)
print('cluster centroids')
for center in model.clusterCenters():
	print(center)
prediction = model.transform(wine_terry)
print('silhouette: %2f' % ClusteringEvaluator(featuresCol='feature').evaluate(prediction))

# four clusters
model = KMeans(featuresCol='feature', k=4).fit(wine_terry)
print('cluster sizes')
print(model.summary.clusterSizes)
print('cluster centroids')
for center in model.clusterCenters():
	print(center)
prediction = model.transform(wine_terry.select('feature'))
print('silhouette: %2f' % ClusteringEvaluator(featuresCol='feature').evaluate(prediction))