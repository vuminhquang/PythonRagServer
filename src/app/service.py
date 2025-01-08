from typing import List, Dict, Optional, Tuple
import asyncio
from app.models import SearchResult
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import torch
import jieba
import re
from rank_bm25 import BM25Okapi
import faiss
import transformers
import warnings
from app.config import settings

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable transformer warnings and batch progress
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add debug logging for settings
logger.info(f"Using CHUNK_SIZE: {settings.CHUNK_SIZE}")
logger.info(f"Using CHUNK_OVERLAP: {settings.CHUNK_OVERLAP}")

class RAGService:
    def __init__(self, 
                 embedding_model: str = settings.EMBEDDING_MODEL,
                 reranker_model: str = settings.RERANKER_MODEL,
                 use_reranker: bool = settings.USE_RERANKER,
                 use_hybrid: bool = settings.USE_HYBRID,
                 device: str = settings.DEVICE):
        logger.info(f"Initializing RAGService with chunk_size={settings.CHUNK_SIZE}")
        self._lock = asyncio.Lock()
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        self.use_hybrid = use_hybrid
        
        self.use_reranker = use_reranker
        if use_reranker:
            logger.info(f"Loading reranker model: {reranker_model}")
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model).to(device)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model if use_reranker else None
        
        self.documents = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.bm25 = None
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using jieba"""
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return list(jieba.cut(text))

    async def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """Add documents to the vector store"""
        if not texts:
            return
            
        logger.info(f"Adding {len(texts)} documents with chunk_size={settings.CHUNK_SIZE}")
        async with self._lock:
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size] if metadatas else None
                
                embeddings = self.embedding_model.encode(batch_texts)
                self.index.add(np.array(embeddings).astype('float32'))
                
                if batch_metadata:
                    self.documents.extend([
                        {"content": text, "metadata": meta}
                        for text, meta in zip(batch_texts, batch_metadata)
                    ])
                else:
                    self.documents.extend([
                        {"content": text, "metadata": {}}
                        for text in batch_texts
                    ])
                
                if self.use_hybrid:
                    for text in batch_texts:
                        self.tokenized_corpus.append(self._tokenize(text))
                
                await asyncio.sleep(0.01)
            
            if self.use_hybrid:
                self.bm25 = BM25Okapi(self.tokenized_corpus)

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for relevant documents"""
        if not self.documents:
            return []

        async with self._lock:
            query_embedding = self.embedding_model.encode([query])[0]
            D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
            
            if not self.use_hybrid:
                # Simple FAISS search
                results = [(self.documents[idx], float(1 / (1 + dist))) 
                          for idx, dist in zip(I[0], D[0])]
            else:
                # Hybrid search with BM25
                tokenized_query = self._tokenize(query)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                faiss_scores = 1 / (1 + D[0])
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)
                
                results = []
                for idx, (faiss_idx, faiss_score) in enumerate(zip(I[0], faiss_scores)):
                    combined_score = 0.5 * faiss_score + 0.5 * bm25_scores[faiss_idx]
                    results.append((self.documents[faiss_idx], combined_score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            if self.use_reranker:
                pairs = [(query, doc["content"]) for doc, _ in results[:top_k]]
                features = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                with torch.no_grad():
                    scores = self.reranker(**features).logits.squeeze()
                    rerank_scores = torch.softmax(scores, dim=0).numpy()
                
                results = [(doc, float(score)) 
                          for (doc, _), score in zip(results[:top_k], rerank_scores)]
            
            return results[:top_k]

    async def clear(self) -> None:
        """Clear all documents"""
        async with self._lock:
            self.documents = []
            self.index = faiss.IndexFlatL2(self.dimension)
            self.bm25 = None
            self.tokenized_corpus = []
