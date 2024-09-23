import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# Inicializando a sessão Spark

def create_spark_session():
    spark = SparkSession.builder \
        .appName("DataTransformation") \
        .getOrCreate()
    return spark

# Função para carregar os datasets

def load_data(spark):
    features_df = spark.read.csv('data/raw/features.csv', header=True, inferSchema=True)
    sales_df = spark.read.csv('data/raw/sales.csv', header=True, inferSchema=True)
    stores_df = spark.read.csv('data/raw/stores.csv', header=True, inferSchema=True)
    return features_df, sales_df, stores_df

# Função para limpeza e transformação dos dados

def clean_and_transform_data(features_df, sales_df, stores_df):

    # Tratamento de valores nulos para features (temperatura e preço de combustível)
    features_df = features_df.fillna(0, subset=['Temperature', 'Fuel_Price'])

    # Converter a coluna de Data
    sales_df = sales_df.withColumn('Date', F.to_date(F.col('Date'), 'yyyy-MM-dd'))

    # Checando e preenchendo valores nulos na tabela de vendas
    sales_df = sales_df.fillna({'Weekly_Sales': 0})

    # Agregando as vendas semanais por loja e calculando média de temperatura por loja
    sales_agg = sales_df.groupBy('Store').agg(
        F.sum('Weekly_Sales').alias('Total_Sales')
    )
    
    # Unindo as vendas agregadas ao dataframe de features para juntar outras informações por Store e Data
    features_sales_df = features_df.join(sales_agg, on='Store', how='inner')

    return features_sales_df, stores_df

# Função para unir os datasets
def join_datasets(features_sales_df, stores_df):
    
    # Unindo os dados de vendas e features com a tabela de lojas
    final_df = features_sales_df.join(stores_df, on='Store', how='inner')

    # Adicionando uma coluna de receita estimada (baseada nas vendas totais e tamanho da loja)
    final_df = final_df.withColumn('Revenue_Estimate', F.col('Total_Sales') * F.col('Size'))

    return final_df

# Função principal
def main():

    # Inicializar a sessão Spark
    spark = create_spark_session()

    # Carregar os datasets
    features_df, sales_df, stores_df = load_data(spark)

    # Limpar e transformar os dados
    features_sales_df, stores_df = clean_and_transform_data(features_df, sales_df, stores_df)

    # Unir os datasets e calcular a receita estimada
    final_df = join_datasets(features_sales_df, stores_df)

    # Salvando o dataframe final em formato Parquet
    final_df.write.mode('overwrite').parquet('data/processed/retail_data_transformed.parquet')

    # Fechar a sessão Spark
    spark.stop()

if __name__ == "__main__":
    main()
