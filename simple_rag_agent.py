import asyncio
import yaml
import requests
from dataclasses import dataclass
from pathlib import Path

import mlflow
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

import chromadb
from chromadb.utils import embedding_functions


# Script directory and config setup
script_dir = Path(__file__).parent
config_path = script_dir / "rag_agent_config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configuration values
model_config = config['model']
oai_model = model_config['full_name']
retries = config['agent']['retries']
system_prompt = config['prompts']['simple_agent_system']

# Embeddings config
embedding_model = config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2')

# ChromaDB setup
CHROMA_PERSIST_DIR = script_dir / "chroma_db"

# Create embedding function (must match what was used during indexing)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_model
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


@dataclass
class Deps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.ClientAPI
    n_results: int = 5  # Number of chunks to retrieve per collection


def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context containing the ChromaDB client.
        search_query: The search query to find relevant documents.
    Returns:
        Retrieved text chunks that are relevant to the search query.
    """
    client = context.deps.chroma_client
    n_results = context.deps.n_results
    
    # Get all collections
    collections = client.list_collections()
    
    if not collections:
        return "No document collections found. Please run init_chromadb.py first."
    
    all_results = []
    
    for collection_name in collections:
        try:
            collection = client.get_collection(
                collection_name, 
                embedding_function=embedding_fn
            )
            
            results = collection.query(
                query_texts=[search_query],
                n_results=n_results
            )
            
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    source = metadata.get('source', collection_name)
                    distance = results['distances'][0][i] if results.get('distances') else None
                    
                    # Format the result
                    result_text = f"[Source: {source}]\n{doc}"
                    all_results.append((distance or 0, result_text))
                    
        except Exception as e:
            print(f"Error querying collection {collection_name}: {e}")
            continue
    
    if not all_results:
        return "No relevant documents found for your query."
    
    # Sort by distance (lower is more relevant) and take top results
    all_results.sort(key=lambda x: x[0])
    top_results = all_results[:n_results]
    
    # Log distances to MLflow span
    distances = [d for d, _ in top_results]
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("retrieval_distances", distances)
        span.set_attribute("retrieval_min_distance", min(distances))
        span.set_attribute("retrieval_mean_distance", sum(distances) / len(distances))
    
    # Format output
    articles = "\n\n---\n\n".join([r[1] for r in top_results])
    # print ("## articles from retreiver ##")
    # print(all_results)
    # print ("## articles from retreiver ##")
    return f"Retrieved {len(top_results)} relevant document sections:\n\n{articles}"



def create_agent() -> Agent:
    """
    Factory function to create a fresh RAG agent.
    Configures the appropriate model provider based on config.
    """
    provider_type = model_config.get('provider', 'openai')
    
    if provider_type == 'vllm':
        # For vLLM, get a fresh API key and use custom base URL
        vllm_api_key = requests.get("http://localhost:8899/access-token").text
        base_url = model_config.get('base_url')
        print(f"Using vLLM provider at: {base_url}")
        
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=vllm_api_key,
        )
        selected_model = OpenAIModel("", provider=provider)
    else:
        # For OpenAI, use the model name directly (uses OPENAI_API_KEY env var)
        selected_model = oai_model
        print(f"Using OpenAI model: {oai_model}")

    the_agent = Agent(
        selected_model,
        retries=retries,
        system_prompt=system_prompt,
        deps_type=Deps,
    )
    
    # Register the retrieve tool
    the_agent.tool(retrieve)
    
    return the_agent


def create_deps(n_results: int = 5) -> Deps:
    """Create dependencies for the RAG agent."""
    return Deps(
        chroma_client=chroma_client,
        n_results=n_results
    )





async def ask(question: str, n_results: int = 5) -> str:
    """
    Ask a question to the RAG agent.
    
    Args:
        question: The question to ask.
        n_results: Number of document chunks to retrieve.
    
    Returns:
        The agent's response.
    """
    # Create a default agent for backwards compatibility
    rag_agent = create_agent()
    deps = create_deps(n_results=n_results)
    result = await rag_agent.run(question, deps=deps)
    return result.output


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if ChromaDB has been initialized
    collections = chroma_client.list_collections()
    if not collections:
        print("No collections found! Please run init_chromadb.py first.")
        sys.exit(1)
    
    print(f"Available collections: {collections}")
    print("-" * 50)
    
    # Example query
    test_question = "What is the percentage breakdown of clinical trial participants by race?"
    print(f"Question: {test_question}\n")
    
    response = asyncio.run(ask(test_question))
    print(f"Answer:\n{response}")

