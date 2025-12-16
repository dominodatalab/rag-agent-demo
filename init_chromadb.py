"""
ChromaDB Initialization Script

This script reads documents from the 'docs' folder and creates a ChromaDB 
collection for each document. Supports PDF, TXT, and MD files.

# First install dependencies
pip install chromadb pypdf
pip install sentence_transformers

# Then run the script
python init_chromadb.py
"""

import os
import sys
import warnings

# Suppress macOS msgtracer warnings from onnxruntime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", message=".*msgtracer.*")

# Suppress ONNX runtime C-level stderr messages
import contextlib
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output (for C-level warnings)."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
import re
from pathlib import Path
import yaml
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# For PDF parsing
try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed. PDF support disabled. Install with: pip install pypdf")


# Load configuration from YAML
script_dir = Path(__file__).parent
config_path = script_dir / "rag_agent_config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Configuration from YAML
DOCS_DIR = script_dir / "docs"
CHROMA_PERSIST_DIR = script_dir / "chroma_db"
EMBEDDING_MODEL = config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2')
CHUNK_SIZE = config.get('embeddings', {}).get('chunk_size', 512)
CHUNK_OVERLAP = config.get('embeddings', {}).get('chunk_overlap', 100)

print(f"Using embedding model: {EMBEDDING_MODEL}")
print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

# Create embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a filename to be a valid ChromaDB collection name.
    ChromaDB collection names must:
    - Be 3-63 characters long
    - Start and end with alphanumeric
    - Contain only alphanumeric, underscores, or hyphens
    - Not contain consecutive periods
    """
    # Remove file extension
    name = Path(name).stem
    
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Ensure it starts and ends with alphanumeric
    name = name.strip('_-')
    
    # Ensure minimum length
    if len(name) < 3:
        name = name + "_doc"
    
    # Truncate if too long
    if len(name) > 63:
        name = name[:63].rstrip('_-')
    
    return name.lower()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    """
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only use if it's not too early
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1
        
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def read_pdf(filepath: Path) -> str:
    """Extract text from a PDF file."""
    if not PDF_SUPPORT:
        raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")
    
    text_parts = []
    with open(filepath, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
    
    return "\n\n".join(text_parts)


def read_text_file(filepath: Path) -> str:
    """Read a text or markdown file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def read_document(filepath: Path) -> str:
    """Read a document based on its file type."""
    suffix = filepath.suffix.lower()
    
    if suffix == '.pdf':
        return read_pdf(filepath)
    elif suffix in ['.txt', '.md', '.markdown']:
        return read_text_file(filepath)
    else:
        print(f"Warning: Unsupported file type '{suffix}' for {filepath.name}")
        return ""


def get_supported_files(docs_dir: Path) -> list[Path]:
    """Get all supported document files from the docs directory."""
    supported_extensions = {'.pdf', '.txt', '.md', '.markdown'}
    files = []
    
    for filepath in docs_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
            files.append(filepath)
    
    return files


def initialize_chromadb():
    """
    Initialize ChromaDB with documents from the docs folder.
    Each document becomes its own collection.
    """
    # Ensure docs directory exists
    if not DOCS_DIR.exists():
        print(f"Creating docs directory at {DOCS_DIR}")
        DOCS_DIR.mkdir(parents=True)
        print("Please add documents to the 'docs' folder and run this script again.")
        return
    
    # Get all supported documents
    doc_files = get_supported_files(DOCS_DIR)
    
    if not doc_files:
        print(f"No supported documents found in {DOCS_DIR}")
        print("Supported formats: PDF, TXT, MD")
        return
    
    print(f"Found {len(doc_files)} document(s) to process")
    
    # Initialize ChromaDB client with persistence
    print(f"\nInitializing ChromaDB at {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Process each document
    for doc_path in doc_files:
        print(f"\n{'='*50}")
        print(f"Processing: {doc_path.name}")
        
        # Create collection name from filename
        collection_name = sanitize_collection_name(doc_path.name)
        print(f"Collection name: {collection_name}")
        
        try:
            # Read document content
            content = read_document(doc_path)
            if not content:
                print(f"  Skipping - no content extracted")
                continue
            
            print(f"  Extracted {len(content)} characters")
            
            # Chunk the content
            chunks = chunk_text(content)
            print(f"  Created {len(chunks)} chunks")
            
            if not chunks:
                print(f"  Skipping - no chunks created")
                continue
            
            # Delete existing collection if it exists (to refresh data)
            try:
                client.delete_collection(collection_name)
                print(f"  Deleted existing collection")
            except ValueError:
                pass  # Collection doesn't exist
            
            # Create new collection with specified embedding function
            collection = client.create_collection(
                name=collection_name,
                metadata={"source_file": doc_path.name},
                embedding_function=embedding_fn
            )
            
            # Add chunks to collection
            ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": doc_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"  ✓ Added {len(chunks)} chunks to collection '{collection_name}'")
            
        except Exception as e:
            print(f"  ✗ Error processing {doc_path.name}: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("ChromaDB Initialization Complete!")
    print(f"Database location: {CHROMA_PERSIST_DIR}")
    
    # List all collections
    collections = client.list_collections()
    print(f"\nCollections created: {len(collections)}")
    for col_name in collections:
        # Get the actual collection object to access count()
        col = client.get_collection(col_name, embedding_function=embedding_fn)
        count = col.count()
        print(f"  - {col_name}: {count} chunks")


def query_collection(collection_name: str, query: str, n_results: int = 3):
    """
    Utility function to query a specific collection.
    Useful for testing after initialization.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    try:
        collection = client.get_collection(collection_name, embedding_function=embedding_fn)
        # Suppress ONNX stderr messages during embedding
        with suppress_stderr():
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
        return results
    except ValueError as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return None


if __name__ == "__main__":
    initialize_chromadb()
    
    # Optional: Test query
    print("\n" + "="*50)
    print("Testing query on available collections...")
    
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collections = client.list_collections()
    
    if collections:
        test_collection_name = collections[0]
        print(f"\nQuerying '{test_collection_name}' with: 'What is this document about?'")
        import sys
        sys.stdout.flush()
        
        results = query_collection(test_collection_name, "What is this document about?")
        
        if results and results['documents'] and results['documents'][0]:
            num_results = len(results['documents'][0])
            print(f"\n✓ Retrieved {num_results} chunks")
            print("\nTop result preview (first 300 chars):")
            print("-" * 40)
            preview = results['documents'][0][0][:300]
            print(preview)
            print("-" * 40)
        else:
            print("\nNo results found.")
        
        print("\n✓ ChromaDB initialization and test complete!")
    else:
        print("\nNo collections found to test.")

