from pathlib import Path

# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Папка с исходными документами
DATA_DIR = BASE_DIR / "data"

# Папка для сохранения векторной базы
CHROMA_DIR = BASE_DIR / "chroma_db"

# Название коллекции в ChromaDB
COLLECTION_NAME = "rag_documents"

# Модель эмбеддингов: поддерживает русский и другие языки
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Настройки разбиения документов
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# Количество наиболее релевантных фрагментов
TOP_K = 3

# Версия конфигурации модели для ответа API
MODEL_VERSION = EMBEDDING_MODEL_NAME

SIMILARITY_THRESHOLD = 0.78
