from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans

# Crear la sesión de Spark
spark = SparkSession.builder.appName("FuzzyCMeans").getOrCreate()

# Definir la URL del archivo CSV
url = "datafinal.csv"

# Leer el archivo CSV y almacenarlo en un DataFrame de PySpark
df_spark = spark.read.option("delimiter", ";").csv(url, header=True, inferSchema=True)

# Convertir las columnas a numéricas y reemplazar las comas por puntos
columns_to_convert = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']
for column in columns_to_convert:
    df_spark = df_spark.withColumn(column, col(column).cast("double").cast("string").cast("double"))

# Seleccionar las columnas relevantes
selected_columns = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']
features = df_spark.select(*selected_columns)

# Crear un objeto StandardScaler para escalar los datos
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Escalar los datos para cada columna
scaler_model = scaler.fit(features)
scaled_features = scaler_model.transform(features)

# Función para realizar Fuzzy C-means
def fuzzy_c_means(data, num_clusters):
    kmeans = KMeans(featuresCol="scaled_features", k=num_clusters, seed=1)
    model = kmeans.fit(data)
    predictions = model.transform(data)
    return predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

# Definir el número de clusters para cada columna
num_clusters_o3 = 3
num_clusters_co = 3
num_clusters_no2 = 3
num_clusters_so2 = 3
num_clusters_pm25 = 3

# Aplicar Fuzzy C-means a cada columna
clusters_o3 = fuzzy_c_means(scaled_features, num_clusters_o3)
clusters_co = fuzzy_c_means(scaled_features, num_clusters_co)
clusters_no2 = fuzzy_c_means(scaled_features, num_clusters_no2)
clusters_so2 = fuzzy_c_means(scaled_features, num_clusters_so2)
clusters_pm25 = fuzzy_c_means(scaled_features, num_clusters_pm25)

# Agregar los resultados al DataFrame original
df_spark = df_spark.withColumn('Cluster_O3', clusters_o3)
df_spark = df_spark.withColumn('Cluster_CO', clusters_co)
df_spark = df_spark.withColumn('Cluster_NO2', clusters_no2)
df_spark = df_spark.withColumn('Cluster_SO2', clusters_so2)
df_spark = df_spark.withColumn('Cluster_PM2_5', clusters_pm25)

# Visualizar los resultados
df_spark.show()

# Guardar el DataFrame 'df_spark' en un archivo de Parquet
df_spark.write.parquet('datafinal_clusters.parquet', mode='overwrite')

# Detener la sesión de Spark
spark.stop()
