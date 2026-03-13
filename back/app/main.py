from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import AskRequest, AnswerResponse, HealthResponse, IngestRequest, IngestResponse, SearchRequest, SearchResponse
from app.services.rag import BaselineRAGService

settings = get_settings()
service = BaselineRAGService(settings=settings)

app = FastAPI(title="RAG Baseline API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(**service.health())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        return service.ingest(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile] = File(...)) -> IngestResponse:
    saved_paths: list[str] = []

    for file in files:
        if not file.filename:
            continue
        content = await file.read()
        destination = service.save_upload(filename=file.filename, content=content)
        saved_paths.append(str(destination))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    return service.ingest(IngestRequest(paths=saved_paths))


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return service.search(query=request.query, limit=request.limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=400, detail="No index available. Ingest documents first.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    try:
        return service.answer(question=request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=400, detail="No index available. Ingest documents first.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
