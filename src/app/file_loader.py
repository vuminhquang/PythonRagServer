import os
from typing import List, Dict
import logging
from pathlib import Path
import asyncio
import PyPDF2
import signal
import sys
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from app.models import Document, DocumentList
from app.service import RAGService
from app.config import settings

# Configure logging and rich console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class FileLoader:
    def __init__(self):
        self.shutdown_flag = False
        
    def handle_interrupt(self, signum=None, frame=None):
        """Handle interrupt signal (Ctrl+C)"""
        self.shutdown_flag = True
        console.print("\n[yellow]Received interrupt signal. Cleaning up...[/yellow]")

    async def setup_signal_handlers(self):
        if sys.platform != 'win32':
            # Unix-like systems
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.handle_interrupt)
        else:
            # Windows
            signal.signal(signal.SIGINT, self.handle_interrupt)
            signal.signal(signal.SIGTERM, self.handle_interrupt)

    def create_chunks(self, text: str, chunk_size: int = None, overlap: int = None, max_chunks: int = 1000) -> List[str]:
        """Create overlapping chunks of text with memory limits"""
        # Use settings from .env, fallback to defaults if not specified
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        text_len = len(text)
        chunk_count = 0

        while start < text_len and chunk_count < max_chunks:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            
            # Skip empty or whitespace-only chunks
            if chunk.strip():
                chunks.append(chunk)
                chunk_count += 1
            
            # Move start position
            start = end - overlap
            
            # Break if we're creating too many chunks
            if chunk_count >= max_chunks:
                logger.warning(f"Reached maximum chunk limit ({max_chunks}) for text")
                break
        
        return chunks

    async def process_pdf(self, file_path: str, progress: Progress, task_id: int) -> List[Dict]:
        """Process a PDF file with simplified progress display"""
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Set total for the task
                progress.update(task_id, total=total_pages, completed=0)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    if self.shutdown_flag:
                        return documents

                    try:
                        text = page.extract_text()
                        if text.strip():
                            text = ' '.join(text.split())
                            chunks = self.create_chunks(text, max_chunks=1000)
                            
                            for chunk_num, chunk in enumerate(chunks, 1):
                                documents.append({
                                    "content": chunk,
                                    "metadata": {
                                        "source": os.path.basename(file_path),
                                        "page": page_num,
                                        "chunk": chunk_num,
                                        "total_pages": total_pages
                                    }
                                })
                                
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        continue
                    
                    # Update progress after processing each page
                    progress.update(task_id, completed=page_num)
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
        
        return documents

    async def scan_upload_folder(self, folder_path: str = "uploads") -> List[Dict]:
        """Scan the upload folder for documents with simplified progress display."""
        documents = []
        
        await self.setup_signal_handlers()
        Path(folder_path).mkdir(exist_ok=True)
        pdf_files = list(Path(folder_path).glob("**/*.pdf"))
        
        if not pdf_files:
            return documents

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            transient=True,
            refresh_per_second=10
        ) as progress:
            try:
                for pdf_file in pdf_files:
                    if self.shutdown_flag:
                        break

                    pdf_task = progress.add_task(
                        f"Processing {pdf_file.name}",
                        total=None,
                        completed=0
                    )
                    
                    pdf_docs = await self.process_pdf(str(pdf_file), progress, pdf_task)
                    documents.extend(pdf_docs)
                    
            except Exception as e:
                logger.error(f"Error details:", exc_info=True)
                raise
            
        return documents

    def _get_all_files(self, folder_path: str = "uploads") -> List[Path]:
        """Get all supported files from the upload directory"""
        Path(folder_path).mkdir(exist_ok=True)
        return list(Path(folder_path).glob("**/*.pdf"))  # Currently only supporting PDFs

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from a file"""
        try:
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    async def auto_load_documents(self, rag_service: RAGService) -> None:
        """Auto-load documents from the upload directory"""
        try:
            logger.info("Starting auto-load documents")
            all_files = self._get_all_files()
            
            if not all_files:
                logger.info("No files found in upload directory")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=True
            ) as progress:
                try:
                    for file_path in all_files:
                        # Create a task for each file
                        task = progress.add_task(
                            f"Processing {file_path.name}",
                            total=None,
                            completed=0
                        )
                        
                        # Use the existing process_pdf method which already handles chunking and page tracking
                        documents = await self.process_pdf(str(file_path), progress, task)
                        
                        if documents:
                            # Add the processed chunks with their metadata
                            await rag_service.add_documents(
                                texts=[doc["content"] for doc in documents],
                                metadatas=[doc["metadata"] for doc in documents]
                            )
                        
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    logger.error("Indexing error details:", exc_info=True)
                    raise
                
        except Exception as e:
            logger.error("Full error details:", exc_info=True) 