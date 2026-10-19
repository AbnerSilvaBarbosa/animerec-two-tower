import os
from pathlib import Path

ROOT = Path(os.getenv("ANIMEREC_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = DATA_DIR / "archive.zip"
ANIME_CSV = RAW_DIR / "anime.csv"
RATING_CSV = RAW_DIR / "rating.csv"
ARTIFACTS = ROOT / "artifacts"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://anime:anime@localhost:5432/anime")

LIKE_THRESHOLD = 8      # nota >= 8 => "gostou" (label 1)
MIN_RATINGS = 20        # usuários com menos avaliações são descartados
TEST_USER_FRACTION = 0.2
HISTORY_FRACTION = 0.5  # metade das notas de cada usuário vira "histórico" (perfil), a outra metade é alvo
EMBEDDING_DIM = 64
