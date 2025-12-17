"""
Web-based chat backend that serves an HTML/JS front-end for the simplest_agent.py philosophy agent.
"""

import argparse
import os
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import mlflow
from domino.agents.tracing import add_tracing
from domino.agents.logging import DominoRun, log_evaluation

# Import the agent factory from simple_rag_agent.py
# We use create_agent() to get a fresh agent with a new API key before each request
# since the VLLM_API_KEY expires every 5 minutes
from simple_rag_agent import create_agent, create_deps, retrieval_distance_accumulator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

# Load configuration from YAML file
config_path = os.path.join(SCRIPT_DIR, 'rag_agent_config.yaml')

app = FastAPI(title="Simple Agent Chat", description="Chat interface for the simple agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """Request model for chat messages"""
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat messages"""
    response: str
    conversation_id: str

@add_tracing(name='single_question_agent', autolog_frameworks=["pydantic_ai"])
async def ask_agent(question):
    # Create a fresh agent with a new API key for each request
    # This handles the 5-minute VLLM_API_KEY expiration
    agent = create_agent()
    deps = create_deps()
    result = await agent.run(question, deps=deps)
    #result = agent.run_sync(question, deps=deps)
    print("*** OUTSIDE SPAN: ****")
    # Get trace ID from the active span
    span = mlflow.get_current_active_span()
    if span: # and retrieval_distance_accumulator:
        print("*** INSIDE SPAN: ****")
        print(retrieval_distance_accumulator)
        trace_id = span.trace_id
        log_evaluation(trace_id=trace_id, name="retrieval_mean_distance", value=sum(retrieval_distance_accumulator) / len(retrieval_distance_accumulator))
        log_evaluation(trace_id=trace_id, name="retrieval_min_distance", value=min(retrieval_distance_accumulator))
        log_evaluation(trace_id=trace_id, name="retrieval_max_distance", value=max(retrieval_distance_accumulator))
        # Clear the accumulator before next query
        retrieval_distance_accumulator.clear()
    return result

@app.post("/chat")
async def chat(request: ChatMessage) -> ChatResponse:
    """
    Process a chat message using the simplest_agent.
    """
    try:
        # Run the agent with the user's message
        with DominoRun(agent_config_path=config_path) as run:
            result = await ask_agent(request.message)

            
        # Generate or use existing conversation ID
        conv_id = request.conversation_id or str(id(request))

        
        return ChatResponse(
            response=result.output,
            conversation_id=conv_id
        )
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "simplest_agent"}


# Serve static files (CSS, JS) - must be after API routes
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the main chat interface"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/{path:path}")
async def serve_static_files(path: str):
    """
    Catch-all route to serve static files.
    This allows the app to work when hosted at any base path.
    """
    file_path = os.path.join(STATIC_DIR, path)
    
    # Security: ensure we don't serve files outside static dir
    if not os.path.abspath(file_path).startswith(os.path.abspath(STATIC_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # For any other path, serve index.html (SPA-style routing)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Simple Agent Chat server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to run the server on (default: 8000 or PORT env var)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    args = parser.parse_args()
    
    log_level = "debug" if args.debug else "info"
    
    print(f"Starting chat server on http://localhost:{args.port}")
    print(f"Serving static files from: {STATIC_DIR}")
    print(f"Log level: {log_level}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=log_level)
