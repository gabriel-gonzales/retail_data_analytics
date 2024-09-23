from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import logging
import sys

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session(app_name='RetailSalesPipeline'):
    """
    Cria e retorna uma sessão Spark.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def read_csv(spark, file_path, schema=None, delimiter=',', header=True):
    """
    Lê um arquivo CSV e retorna um DataFrame Spark.
    """
    logger.info(f"Lendo arquivo CSV: {file_path}")
    
    if schema:
        df = spark.read \
            .option("header", header) \
            .option("delimiter", delimiter) \
            .schema(schema) \
            .csv(file_path)
    else:
        df = spark.read \
            .option("header", header) \
            .option("delimiter", delimiter) \
            .option("inferSchema", "true") \
            .csv(file_path)
    
    return df

def validate_dataframe(df, df_name):
    """
    Realiza validações básicas no DataFrame:
    - Contagem inicial de linhas
    - Remoção de linhas com valores nulos
    - Remoção de duplicatas
    """
    logger.info(f"Validando DataFrame: {df_name}")
    
    initial_count = df.count()
    logger.info(f"Número inicial de linhas: {initial_count}")
    
    # Remover linhas com valores nulos
    df_cleaned = df.na.drop()
    cleaned_count = df_cleaned.count()
    logger.info(f"Número de linhas após remoção de nulos: {cleaned_count}")
    
    # Remover duplicatas
    df_cleaned = df_cleaned.dropDuplicates()
    dedup_count = df_cleaned.count()
    logger.info(f"Número de linhas após remoção de duplicatas: {dedup_count}")
    
    return df_cleaned

def save_parquet(df, output_path):
    """
    Salva o DataFrame em formato Parquet.
    """
    logger.info(f"Salvando DataFrame em Parquet: {output_path}")
    df.write.mode("overwrite").parquet(output_path)

def main():
    try:
        # Inicializar Spark
        spark = create_spark_session()
        
        # Caminhos dos arquivos brutos
        raw_data_path = "data/raw/"
        
        # Leitura dos datasets
        sales_df = read_csv(spark, f"{raw_data_path}sales.csv")
        features_df = read_csv(spark, f"{raw_data_path}features.csv")
        stores_df = read_csv(spark, f"{raw_data_path}stores.csv")
        
        # Validação dos datasets
        sales_df_clean = validate_dataframe(sales_df, "Sales Data")
        features_df_clean = validate_dataframe(features_df, "Features Data")
        stores_df_clean = validate_dataframe(stores_df, "Stores Data")
        
        # Salvando os datasets processados
        save_parquet(sales_df_clean, "data/processed/sales_data.parquet") 
        save_parquet(features_df_clean, "data/processed/features_data.parquet")
        save_parquet(stores_df_clean, "data/processed/stores_data.parquet")
        
        logger.info("Ingestão de dados concluída com sucesso.")
        
        # Encerrar a sessão Spark
        spark.stop()
        
    except Exception as e:
        logger.error("Ocorreu um erro durante a ingestão de dados.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
