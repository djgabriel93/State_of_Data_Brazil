from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_ORIGINAIS = PASTA_DADOS / "State_of_Data_2024.csv"
DADOS_TRATADO = PASTA_DADOS / "State_of_Data_2024_tratado.parquet"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PASTA_PROJETO / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"
