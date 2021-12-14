import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df_terry = spark.read.csv('./data/retail-data/by-day/2010-12-02.csv', header=True, inferSchema=True)
print(df_terry.printSchema())
df_terry.cache()
print(df_terry.count())
print(df_terry.select('CustomerID').distinct().count())
print(df_terry.select('country').distinct().count())
pd.set_option('display.max_columns', 100)
print(df_terry.filter("country == 'United Kingdom' and quantity > 300").toPandas())
print(df_terry.filter("description rlike 'CHOCOLATE'").toPandas())