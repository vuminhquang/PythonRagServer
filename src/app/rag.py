import faiss
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from dataclasses import dataclass

@dataclass
class Document:
    """Document class to store content and metadata"""
    content: str
    metadata: Dict = None

class RAG:
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-base-zh",
                 reranker_model: Optional[str] = None,
                 device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize models
        self.encoder = SentenceTransformer(embedding_model, device=self.device)
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None
        
        # Initialize FAISS index
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Cache for documents and embeddings
        self.documents = []
        self.embeddings_cache = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = self.encoder.encode(text, convert_to_numpy=True)
        return self.embeddings_cache[text]

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the index"""
        if not texts:
            return
            
        # Process embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encoder.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.extend(batch_embeddings)
            
            # Cache embeddings
            for text, emb in zip(batch_texts, batch_embeddings):
                self.embeddings_cache[text] = emb

        embeddings = np.vstack(all_embeddings)
        self.index.add(embeddings)
        
        # Store documents
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.documents.extend(list(zip(texts, metadatas)))

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for relevant documents"""
        query_embedding = self._get_embedding(query)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k=k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc_text, metadata = self.documents[idx]
                
                if self.reranker:
                    # Rerank score
                    rerank_score = self.reranker.predict([[query, doc_text]])
                    score = float(rerank_score)
                
                results.append(({"content": doc_text, "metadata": metadata}, float(score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def clear(self):
        """Clear the index and cached documents"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents.clear()
        self.embeddings_cache.clear() 