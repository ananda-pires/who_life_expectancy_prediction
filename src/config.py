"""
Configurações e caminhos do projeto.
"""

from pathlib import Path

# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[1]

# Pastas de dados
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FINAL_DATA_DIR = BASE_DIR / "data" / "final"

# Saídas
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Modelos
MODELS_DIR = BASE_DIR / "models" / "v1"

MODEL_FILE = MODELS_DIR / "modelo_regressao_v1.pkl"
METRICS_FILE = MODELS_DIR / "metricas_v1.json"

# Arquivo bruto baixado da API da WHO
RAW_DATA_FILE = RAW_DATA_DIR / "who_health_inequality_data.xlsx"

# Garante que as pastas existam
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)