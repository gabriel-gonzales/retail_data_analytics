import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# Inicializando a sessão Spark
def create_spark_session():
    spark = SparkSession.builder \
        .appName("DataTransformation") \
        .getOrCreate()
    return spark

# Carregar os datasets
def load_data(spark):
    features_df = spark.read.csv('data/raw/features.csv', header=True, inferSchema=True)
    sales_df = spark.read.csv('data/raw/sales.csv', header=True, inferSchema=True)
    stores_df = spark.read.csv('data/raw/stores.csv', header=True, inferSchema=True)
    return features_df, sales_df, stores_df

# Limpeza e transformação dos dados
def clean_and_transform_data(features_df, sales_df):
    # Tratamento de valores nulos
    features_df = features_df.fillna(0, subset=['Temperature', 'Fuel_Price'])

    # Transformando os tipos de dados (data)
    sales_df = sales_df.withColumn('Date', F.to_date(F.col('Date'), 'MM/dd/yyyy'))

    return features_df, sales_df

# Agregação (soma das vendas semanais por loja)
def aggregate_sales(sales_df):
    sales_agg = sales_df.groupBy('Store').agg(
        F.sum('Weekly_Sales').alias('Total_Sales')
    )
    return sales_agg

# Unindo os dados (Join)

def join_datasets(features_df, sales_agg, stores_df):
    # Unir as vendas agregadas com os dados das lojas
    sales_stores_df = sales_agg.join(stores_df, on='Store', how='inner')

    # Unir o resultado com os features (inclui Temperature)
    final_df = sales_stores_df.join(features_df, on='Store', how='inner')

    return final_df

# Função principal
def main():

    # Inicializar a sessão Spark
    spark = create_spark_session()

    # Carregar os datasets
    features_df, sales_df, stores_df = load_data(spark)

    # Limpar e transformar os dados
    features_df, sales_df = clean_and_transform_data(features_df, sales_df)

    # Agregar as vendas
    sales_agg = aggregate_sales(sales_df)

    # Unir os datasets
    final_df = join_datasets(features_df, sales_agg, stores_df)

    # Cálculo da média de temperatura após a junção
    final_df = final_df.groupBy('Store').agg(
        F.sum('Total_Sales').alias('Total_Sales'),
        F.avg('Temperature').alias('Avg_Temperature')
    )

    # Exemplo de salvamento do dataframe final
    final_df.write.mode('overwrite').parquet('data/processed/retail_data_transformed.parquet')

    # Fechar a sessão Spark
    spark.stop()

if __name__ == "__main__":
    main()
