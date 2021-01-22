
# Clément Couvert


from pyspark.sql import SparkSession
import configparser
from pyspark.sql.functions import col
import folium
import pandas as pd
import webbrowser

# 1
spark = SparkSession.builder \
    .master("local") \
    .appName("Bristol") \
    .getOrCreate()

# 2
config = configparser.ConfigParser()
config.read('properties.conf')
path_to_input_data = config['Bristol-City-bike']['Input-data']
path_to_output_data = config['Bristol-City-bike']['Output-data']
num_partition_kmeans = config.getint('Bristol-City-bike', 'Kmeans-level')

# 3
bristol = spark.read.json(path_to_input_data)
bristol.show(3)

#4
Kmean_df = bristol.select(col("latitude"), col("longitude"))
Kmean_df.show(3)

# 5
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

features = ("longitude", "latitude")
kmeans = KMeans().setK(num_partition_kmeans).setSeed(1)
assembler = VectorAssembler(inputCols=features, outputCol="features")
dataset = assembler.transform(Kmean_df)
model = kmeans.fit(dataset)
fitted = model.transform(dataset)

# 6
fitted.columns


# 7
fitted.groupBy("prediction") \
    .mean() \
    .show()

fitted.createOrReplaceTempView("Fit")

spark.sql(
    """select Avg(longitude) as Moy_Long,  Avg(latitude) as Moy_lat, Prediction From Fit Group By prediction""").show()
