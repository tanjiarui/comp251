import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()
df_terry = spark.read.csv(['./data/retail-data/by-day/2011-01-09.csv', './data/retail-data/by-day/2011-01-10.csv'], header=True)
print(df_terry.count())
print(df_terry.printSchema())
# stock id start with 227 and description start with alarm clock or unite price greater than 5
df2_terry = df_terry.filter("StockCode rlike '227' and Description rlike 'ALARM CLOCK' or UnitPrice > 5").withColumn('Quantity', df_terry['Quantity'].cast(IntegerType()))
print(df2_terry.groupby().sum('Quantity').collect()[0][0])
print(df2_terry.groupby().min('Quantity').collect()[0][0])
print(df2_terry.groupby().max('Quantity').collect()[0][0])
# not from uk
pd.set_option('display.max_columns', 100)
print(df2_terry.filter("Country != 'United Kingdom'").toPandas())