from pydantic_settings import BaseSettings
from typing import Literal
import torch
import logging
from dotenv import load_dotenv

# Load .env file first - this is needed!
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKER: bool = True
    USE_HYBRID: bool = False
    USE_GPU: bool = False
    DEVICE: str = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"

    # Search settings
    CHUNK_SIZE: int = 10000  # Very large chunk size to keep entire page
    CHUNK_OVERLAP: int = 0    # No overlap between pages
    MIN_SCORE: float = 0.5  
    INITIAL_POOL_MULTIPLIER: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 