import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Inicializando a sessão Spark
def create_spark_session():
    spark = SparkSession.builder \
        .appName("DataAnalysis") \
        .getOrCreate()
    return spark

# Carregar os dados transformados
def load_transformed_data(spark):
    transformed_df = spark.read.parquet('data/processed/retail_data_transformed.parquet')
    return transformed_df

# Análise 1: Resumo Estatístico das Vendas
def summary_statistics(transformed_df):
    print("---- Resumo Estatístico das Vendas ----")
    stats_df = transformed_df.describe(['Total_Sales', 'Revenue_Estimate'])
    stats_pd = stats_df.toPandas().set_index('summary')
    return stats_pd.to_html()

# Análise 2: Vendas por Tipo de Loja
def sales_by_store_type(transformed_df):
    print("---- Vendas por Tipo de Loja ----")
    sales_type_df = transformed_df.groupBy('Type').agg(
        F.sum('Total_Sales').alias('Total_Sales'),
        F.avg('Total_Sales').alias('Avg_Sales')
    ).orderBy('Total_Sales', ascending=False)
    sales_type_df.show()
    
    # Coletar para visualização
    sales_type_pd = sales_type_df.toPandas()
    
    # Plotar Vendas Totais por Tipo de Loja
    plt.figure(figsize=(10,6))
    sns.barplot(x='Type', y='Total_Sales', data=sales_type_pd)
    plt.title('Vendas Totais por Tipo de Loja')
    plt.xlabel('Tipo da Loja')
    plt.ylabel('Vendas Totais')
    plt.tight_layout()
    plt.savefig('output/plots/sales_by_store_type.png')
    plt.close()

# Análise 3: Impacto do Tamanho da Loja nas Vendas
def sales_vs_store_size(transformed_df):
    print("---- Impacto do Tamanho da Loja nas Vendas ----")
    sales_size_df = transformed_df.select('Size', 'Total_Sales').sample(fraction=0.1).toPandas()
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='Size', y='Total_Sales', data=sales_size_df)
    plt.title('Relação entre Tamanho da Loja e Vendas Totais')
    plt.xlabel('Tamanho da Loja')
    plt.ylabel('Vendas Totais')
    plt.tight_layout()
    plt.savefig('output/plots/sales_vs_store_size.png')
    plt.close()

# Análise 4: Vendas ao Longo do Tempo
def sales_over_time(transformed_df):
    print("---- Vendas Totais ao Longo do Tempo (Agrupadas por Mês) ----")

    # Adicionar colunas de Ano e Mês
    transformed_df = transformed_df.withColumn('Year', F.year('Date'))
    transformed_df = transformed_df.withColumn('Month', F.month('Date'))
    
    # Agrupar por Ano e Mês e somar as vendas
    monthly_sales_df = transformed_df.groupBy('Year', 'Month').agg(
        F.sum('Total_Sales').alias('Total_Sales')
    ).orderBy('Year', 'Month')
    
    # Criar uma coluna de Data para facilitar a plotagem (primeiro dia do mês)
    monthly_sales_df = monthly_sales_df.withColumn('Month_Start', F.to_date(F.concat_ws('-', F.col('Year'), F.col('Month'), F.lit('01')), 'yyyy-M-d'))
    
    # Converter para Pandas para visualização
    monthly_sales_pd = monthly_sales_df.select('Month_Start', 'Total_Sales').toPandas()

    # Certificar que 'Month_Start' está no formato datetime
    monthly_sales_pd['Month_Start'] = pd.to_datetime(monthly_sales_pd['Month_Start'])
    
    # Configurar o estilo do Seaborn
    sns.set(style="whitegrid")
    
    # Plotar as vendas mensais
    plt.figure(figsize=(14,7))
    sns.lineplot(x='Month_Start', y='Total_Sales', data=monthly_sales_pd, marker='o')
    plt.title('Vendas Totais Mensais ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/plots/sales_over_time_monthly.png')
    plt.close()

    # Adicionar Média Móvel (por exemplo, 3 meses)
    monthly_sales_pd['Rolling_Avg_Sales'] = monthly_sales_pd['Total_Sales'].rolling(window=3).mean()

    # Plotar Vendas Totais e Média Móvel
    plt.figure(figsize=(14,7))
    sns.lineplot(x='Month_Start', y='Total_Sales', data=monthly_sales_pd, marker='o', label='Vendas Totais')
    sns.lineplot(x='Month_Start', y='Rolling_Avg_Sales', data=monthly_sales_pd, marker='o', label='Média Móvel (3 Meses)')
    plt.title('Vendas Totais Mensais com Média Móvel')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/plots/sales_over_time_monthly_with_rolling_avg.png')
    plt.close()

    # Filtrar os últimos 2 anos (ajuste conforme necessário)
    latest_year = monthly_sales_pd['Month_Start'].dt.year.max()
    filtered_sales_pd = monthly_sales_pd[monthly_sales_pd['Month_Start'].dt.year >= (latest_year - 2)]

    # Plotar
    plt.figure(figsize=(14,7))
    sns.lineplot(x='Month_Start', y='Total_Sales', data=filtered_sales_pd, marker='o')
    plt.title('Vendas Totais Mensais dos Últimos 2 Anos')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/plots/sales_over_time_last_2_years.png')
    plt.close()



# Análise 5: Impacto de Feriados nas Vendas
def sales_impact_holidays(transformed_df):
    print("---- Impacto de Feriados nas Vendas ----")
    holiday_df = transformed_df.groupBy('IsHoliday').agg(
        F.sum('Total_Sales').alias('Total_Sales'),
        F.avg('Total_Sales').alias('Avg_Sales')
    )
    holiday_df.show()
    
    # Coletar para visualização
    holiday_pd = holiday_df.toPandas()
    
    # Plotar Vendas em Feriados vs Não Feriados
    plt.figure(figsize=(8,6))
    sns.barplot(x='IsHoliday', y='Total_Sales', data=holiday_pd)
    plt.title('Vendas Totais: Feriados vs Não Feriados')
    plt.xlabel('É Feriado')
    plt.ylabel('Vendas Totais')
    plt.tight_layout()
    plt.savefig('output/plots/sales_impact_holidays.png')
    plt.close()

# Análise 6: Correlação entre Variáveis
def correlation_analysis(transformed_df):
    print("---- Análise de Correlação ----")
    # Selecionar colunas numéricas para correlação
    numeric_df = transformed_df.select('Total_Sales', 'Revenue_Estimate', 'Size', 'Temperature', 'Fuel_Price')
    
    # Coletar para análise de correlação
    numeric_pd = numeric_df.toPandas()
    
    # Calcular a matriz de correlação
    correlation = numeric_pd.corr()
    print(correlation)
    
    # Plotar a matriz de correlação
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação entre Variáveis')
    plt.tight_layout()
    plt.savefig('output/plots/correlation_matrix.png')
    plt.close()

# Gerar o Relatório HTML
def generate_html_report(summary_stats):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('output/reports/report_template.html')
    
    # Renderizar o template com as estatísticas
    html_content = template.render(summary_statistics=summary_stats)
    
    # Salvar o relatório
    with open('output/reports/analysis_report.html', 'w') as f:
        f.write(html_content)

# Função principal
def main():
    
    # Inicializar a sessão Spark
    spark = create_spark_session()
    
    # Carregar os dados transformados
    transformed_df = load_transformed_data(spark)
    
    # Executar análises e coletar estatísticas
    summary_stats = summary_statistics(transformed_df)
    sales_by_store_type(transformed_df)
    sales_vs_store_size(transformed_df)
    sales_over_time(transformed_df)
    sales_impact_holidays(transformed_df)
    correlation_analysis(transformed_df)
    
    # Gerar o relatório HTML
    generate_html_report(summary_stats)
    
    # Fechar a sessão Spark
    spark.stop()

if __name__ == "__main__":
    main()
