from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import min, desc

spark = SparkSession.builder.getOrCreate()
spark.conf.set('spark.sql.shuffle.partitions', 6)

vertex = spark.createDataFrame([
	('Yara', 15, 'student'),
	('Ali', 64, 'engineer'),
	('Mark', 32, 'doctor'),
	('Maisa', 58, 'teacher'),
	('Nora', 35, 'teacher'),
	('Bob', 45, 'engineer'),
	('Parth', 14, 'student'),
	('Mary', 8, 'student')
], ['id', 'age', 'occupation'])

edges = spark.createDataFrame([
	('Yara', 'Ali', 'daughter'),
	('Ali', 'Mark', 'follows'),
	('Ali', 'Maisa', 'married'),
	('Maisa', 'Ali', 'married'),
	('Ali', 'Bob', 'brothers'),
	('Bob', 'Ali', 'brothers'),
	('Mary', 'Bob', 'daughter'),
	('Parth', 'Bob', 'son'),
	('Parth', 'Maisa', 'follows'),
	('Parth', 'Nora', 'follows')
], ['src', 'dst', 'relationship'])

terry_graph = GraphFrame(vertex, edges)
terry_graph.cache()
terry_graph.vertices.select(min('age')).show()
terry_graph.vertices.filter("occupation == 'engineer'").show()
terry_graph.edges.groupby('src').count().orderBy(desc('count')).show(2)
scc = terry_graph.stronglyConnectedComponents(maxIter=10)
scc.groupBy('component').count().show()
scc.where('component == 0').show()