from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from app.models import DocumentList, Query, SearchResponse, SearchResult
from app.service import RAGService
from app.config import settings
from app.file_loader import FileLoader
import logging
from pathlib import Path as PathLib
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FastRAG API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService(
    embedding_model=settings.EMBEDDING_MODEL,
    reranker_model=settings.RERANKER_MODEL,
    use_reranker=settings.USE_RERANKER,
    device=settings.DEVICE
)

file_loader = FileLoader()

@app.on_event("startup")
async def startup_event():
    """Load documents from upload folder when the server starts"""
    logger.info("Loading documents from upload folder...")
    try:
        await file_loader.auto_load_documents(rag_service)
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

@app.post("/documents", status_code=201)
async def add_documents(documents: DocumentList):
    """Add documents to the RAG system"""
    try:
        logger.info(f"Adding {len(documents.documents)} documents")
        await rag_service.add_documents(
            texts=[doc.content for doc in documents.documents],
            metadatas=[doc.metadata for doc in documents.documents]
        )
        return {"message": f"Successfully added {len(documents.documents)} documents"}
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_documents():
    """Reload all documents from the upload folder"""
    try:
        await rag_service.clear()
        await file_loader.auto_load_documents(rag_service)
        return {"message": "Successfully reloaded all documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: Query) -> SearchResponse:
    """Search for relevant documents"""
    try:
        # Use the raw query text directly
        raw_results = await rag_service.search(query.text, query.top_k)
        
        # Debug logging
        logger.info("Raw search results:")
        logger.info(f"Type: {type(raw_results)}")
        logger.info(f"Content: {raw_results}")
        
        # Convert the raw results into SearchResult objects with enhanced metadata
        results = []
        for result in raw_results:
            content = result[0]["content"]
            metadata = result[0]["metadata"]
            score = float(result[1])
            
            # Enhance the result with file information
            file_path = PathLib("uploads") / metadata["source"]
            enhanced_metadata = {
                **metadata,  # Keep existing metadata (source, page, chunk, total_pages)
                "file_info": {
                    "size": file_path.stat().st_size if file_path.exists() else None,
                    "modified": file_path.stat().st_mtime if file_path.exists() else None,
                    "full_path": str(file_path)
                }
            }
            
            # Create the search result
            results.append(SearchResult(
                content=content,  # The chunk content
                metadata=enhanced_metadata,
                score=score
            ))
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear():
    """Clear all documents from the RAG system"""
    try:
        await rag_service.clear()
        return {"message": "Successfully cleared all documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Retrieve a specific document's content"""
    try:
        # Assuming document_id is the filename in the uploads folder
        file_path = PathLib("uploads") / document_id
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
            
        if file_path.suffix.lower() != '.pdf':
            raise HTTPException(status_code=400, detail="Unsupported file type")
            
        # Return the document metadata and content
        text = file_loader._extract_text(file_path)
        return {
            "id": document_id,
            "content": text,
            "metadata": {
                "source": str(file_path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all available documents"""
    try:
        files = file_loader._get_all_files()
        return {
            "documents": [
                {
                    "id": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                } for f in files
            ]
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
