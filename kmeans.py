from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

# Crear una sesión de Spark
spark = SparkSession.builder.appName('kmeans').getOrCreate()

# Leer el archivo CSV y almacenarlo en un DataFrame de Spark
df = spark.read.format('csv').option('header', 'true').option('sep', ';').load('datafinal.csv')

# Convertir las columnas a numéricas
for column in ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']:
    df = df.withColumn(column, col(column).cast('float'))

# Lista de columnas a escalar
columns_to_scale = ['O3', 'CO', 'NO2', 'SO2', 'PM2_5']

# Escalar las columnas
for column in columns_to_scale:
    assembler = VectorAssembler(inputCols=[column], outputCol=column+"_vec")
    scaler = StandardScaler(inputCol=column+"_vec", outputCol=column+"_scaled")
    df = assembler.transform(df)
    df = scaler.fit(df).transform(df)

# Aplicar KMeans a cada columna
for column in columns_to_scale:
    kmeans = KMeans(featuresCol=column+'scaled', predictionCol='Cluster'+column, k=3)
    model = kmeans.fit(df)
    df = model.transform(df)

# Visualizar los resultados
df.show()

# Guardar el DataFrame 'df' en un archivo de Excel
df.write.format('com.databricks.spark.csv').option('header', 'true').save('datafinal_clusters.csv')