from pathlib import Path
import tempfile
from pydantic import BaseModel, Field
from app.services.agent_service import run_agent
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.document_processor import (
    extract_text,
    split_into_chunks,
)
from app.services.vector_store import store_document_chunks
from app.services.rag_service import answer_question


router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@router.get("/")
def root():
    return {
        "message": "Enterprise Knowledge Copilot API is running",
        "status": "success",
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Enterprise Knowledge Copilot",
    }


@router.get("/api/info")
def application_info():
    return {
        "application": "AI Enterprise Knowledge Copilot",
        "version": "1.0.0",
        "features": [
            "Document Processing",
            "Embeddings",
            "Qdrant Vector Search",
            "RAG Question Answering",
        ],
    }


@router.post("/api/documents/process")
async def process_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must be 10 MB or less.",
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        text = extract_text(str(temporary_path))

        chunks = split_into_chunks(
            text,
            chunk_size=500,
            overlap=50,
        )

        stored_count = store_document_chunks(
            chunks,
            filename,
        )

        return {
            "status": "success",
            "filename": filename,
            "characters": len(text),
            "words": len(text.split()),
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "extracted_text": text,
            "chunks": chunks,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Document processing error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Document processing or indexing failed.",
        ) from error

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


@router.post("/api/chat")
def chat_with_documents(request: QuestionRequest):
    try:
        return answer_question(
            request.question,
            top_k=request.top_k,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"RAG error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not generate an answer. Check server logs.",
        ) from error
        
class AgentRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        max_length=30,
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
    )


@router.post("/api/agent")
def agent_endpoint(request: AgentRequest):
    try:
        return run_agent(
            task=request.task,
            question=request.question,
            top_k=request.top_k,
        )

    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail="Agent workflow failed. Please try again.",
        ) from exc