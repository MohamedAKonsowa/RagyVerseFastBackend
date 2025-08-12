import os
import json
import time
import traceback
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Optional, List, Tuple

import fitz  # PyMuPDF
import networkx as nx
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from pydub import AudioSegment

from elevenlabs.client import ElevenLabs
from openai import OpenAI

# LangChain bits kept (vector store + docs)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------
# ENV & APP
# -----------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="RagyVerse API (GPT-powered)", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature flags / knobs
ENABLE_OCR = os.getenv("ENABLE_OCR", "0") == "1"  # default OFF for fast boot
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))

# -----------------------------
# API KEYS / CLIENTS
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
oai_client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # e.g., "gpt-4o", "gpt-4o-mini", "gpt-5"

ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
if not ELEVEN_API_KEY or not VOICE_ID:
    raise RuntimeError("Missing ELEVEN_LABS_API_KEY or VOICE_ID")
client_11labs = ElevenLabs(api_key=ELEVEN_API_KEY)

# -----------------------------
# GLOBAL STATE (lazy singletons)
# -----------------------------
document_text = ""
knowledge_graph: Optional[nx.DiGraph] = None
graph_triples: List[tuple] = []
retriever = None

# Lazy caches
_spacy_nlp = None
_embeddings = None
_ocr_pipe = None
_reranker = None


def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def get_embeddings():
    """
    Lazy-load embeddings. Default: MiniLM (already in your env).
    You can switch to BGE with EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    """
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        kwargs = {}
        # If using BGE, normalize embeddings improves cosine search
        if "bge" in model_name.lower():
            kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        _embeddings = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    return _embeddings


def get_ocr():
    """
    Lazy-load OCR vision model (only if ENABLE_OCR=1).
    Default: BLIP captioning (works on many PDFs but heavy).
    """
    global _ocr_pipe
    if _ocr_pipe is None:
        from transformers import pipeline
        model_name = os.getenv("OCR_MODEL", "Salesforce/blip-image-captioning-base")
        _ocr_pipe = pipeline("image-to-text", model=model_name)
    return _ocr_pipe


def get_reranker():
    """
    Lazy-load a lightweight cross-encoder reranker for better top-k.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# -----------------------------
# GPT helper
# -----------------------------
def call_gpt(system_prompt: str, user_prompt: str) -> str:
    """
    Uses OpenAI Responses API; returns text.
    """
    resp = oai_client.responses.create(
        model=GPT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.2,  # uncomment/tune if needed
        # max_output_tokens=600,
    )
    try:
        return resp.output_text.strip()
    except Exception:
        return (getattr(resp, "output", "") or "").strip() or "No answer returned."


# -----------------------------
# PDF → Text (optional OCR)
# -----------------------------
def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract selectable text; if none and OCR is enabled, run OCR on images.
    """
    try:
        temp_path = NamedTemporaryFile(delete=False, suffix=".pdf").name
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        doc = fitz.open(temp_path)
        extracted: List[str] = []

        for page in doc:
            text = page.get_text("text") or ""
            if text.strip():
                extracted.append(text)
                continue

            # If no text and OCR enabled, parse page images
            if ENABLE_OCR:
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    image = doc.extract_image(xref)
                    image_bytes = image["image"]
                    try:
                        res = get_ocr()(image_bytes)
                        if res:
                            gen = res[0].get("generated_text") or res[0].get("caption") or ""
                            if gen:
                                extracted.append(gen)
                    except Exception:
                        # skip OCR errors per image
                        pass

        os.unlink(temp_path)
        return "\n".join(extracted).strip() if extracted else "⚠️ No extractable text found"
    except Exception as e:
        return f"❌ Error processing PDF: {e}"


# -----------------------------
# Simple Knowledge Graph
# -----------------------------
def build_knowledge_graph(text: str):
    """
    Naive KG: connect first/last non-stopword tokens per sentence.
    """
    global knowledge_graph, graph_triples
    knowledge_graph = nx.DiGraph()
    graph_triples = []

    try:
        doc = get_spacy()(text)
        for sent in doc.sents:
            tokens = [t.text for t in sent if not t.is_stop and not t.is_punct]
            if len(tokens) >= 3:
                subj, rel, obj = tokens[0], "related_to", tokens[-1]
                knowledge_graph.add_edge(subj, obj, relation=rel)
                graph_triples.append((subj, rel, obj))
    except Exception:
        pass

    return graph_triples


def query_knowledge_graph(query: str):
    if not knowledge_graph:
        return []
    q = (query or "").lower()
    return [
        (u, data["relation"], v)
        for u, v, data in knowledge_graph.edges(data=True)
        if q in u.lower() or q in v.lower()
    ]


# -----------------------------
# Retrieval (FAISS) + Enhancements
# -----------------------------
def initialize_retriever(text: str):
    """
    Build FAISS index and create a retriever (invoke-based).
    """
    global retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents([Document(page_content=text)])

    # attach metadata for citations
    docs = []
    for i, d in enumerate(chunks):
        d.metadata = {"chunk_id": i}
        docs.append(d)

    vectorstore = FAISS.from_documents(docs, get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})  # deeper pool; we'll rerank


def expand_queries(query: str) -> List[str]:
    """
    Multi-Query + HyDE via GPT. Returns a list of expanded queries.
    """
    system = "You expand search queries for retrieval."
    prompt = (
        "Original question:\n"
        f"{query}\n\n"
        "1) Provide 3 diverse paraphrases that might retrieve complementary evidence.\n"
        "2) Provide 1 short hypothetical answer (HyDE) capturing likely key terms.\n"
        'Return strict JSON: {"paraphrases": [...], "hyde": "..."}'
    )
    out = call_gpt(system, prompt)
    try:
        data = json.loads(out)
        paras = [query] + list({p.strip() for p in data.get("paraphrases", []) if p and p.strip()})
        hyde = (data.get("hyde") or "").strip()
        return paras + ([hyde] if hyde else [])
    except Exception:
        return [query]


def retrieve_pool(expanded_queries: List[str]) -> List[Document]:
    """
    Use retriever.invoke for each expanded query and merge unique docs.
    """
    if not retriever:
        return []
    pool: List[Document] = []
    seen = set()
    for q in expanded_queries:
        docs = retriever.invoke(q)  # new API; replaces get_relevant_documents
        for d in docs:
            key = (d.metadata.get("chunk_id"), d.page_content[:150])
            if key not in seen:
                pool.append(d)
                seen.add(key)
    return pool


def rerank_docs(query: str, docs: List[Document], top_k: int = TOP_K_RERANK) -> List[Document]:
    if not docs:
        return []
    rr = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = rr.predict(pairs)  # higher is better
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def pack_context_with_citations(docs: List[Document]) -> Tuple[str, List[dict]]:
    snippets = []
    cites: List[dict] = []
    for idx, d in enumerate(docs, 1):
        snippet = (d.page_content or "").strip()[:1200]  # keep tight to avoid LLM drift
        cid = d.metadata.get("chunk_id", idx)
        snippets.append(f"[{idx}] {snippet}")
        cites.append({"id": idx, "chunk_id": cid})
    return "\n\n".join(snippets), cites


# -----------------------------
# TTS + STT
# -----------------------------
def generate_tts(text: str) -> tuple:
    """
    ElevenLabs TTS -> mp3 + wav (for Unity).
    """
    audio_bytes = b"".join(
        client_11labs.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
    )

    os.makedirs("static", exist_ok=True)
    os.makedirs("saved_outputs", exist_ok=True)

    filename_base = f"response_{int(time.time())}"
    mp3_path = os.path.join("static", filename_base + ".mp3")
    wav_path = os.path.join("saved_outputs", "latest_output.wav")

    with open(mp3_path, "wb") as f:
        f.write(audio_bytes)

    sound = AudioSegment.from_file(mp3_path, format="mp3")
    sound.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

    return mp3_path, wav_path


def transcribe_audio(file: UploadFile) -> str:
    """
    ElevenLabs Scribe (STT). Returns text or an error string starting with '❌'.
    """
    try:
        audio_data = BytesIO(file.file.read())
        transcription = client_11labs.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1",
            tag_audio_events=True,
            language_code="eng",
            diarize=True,
        )
        return getattr(transcription, "text", None) or str(transcription)
    except Exception as e:
        return f"❌ Error transcribing: {e}"


# -----------------------------
# Chat endpoint core
# -----------------------------
def chat_with_bot(query: str):
    """
    Enhanced RAG: expand → retrieve pool → rerank → cite → GPT → TTS
    """
    global document_text, retriever
    if not document_text:
        return {"error": "❌ Upload a PDF first"}, None
    if not retriever:
        initialize_retriever(document_text)

    try:
        # 1) Expand queries (Multi-Query + HyDE)
        expanded = expand_queries(query)

        # 2) Retrieve many candidates
        pool = retrieve_pool(expanded)

        # 3) Rerank to top-k
        top_docs = rerank_docs(query, pool, top_k=TOP_K_RERANK)

        # 4) Build compact context + citations
        context_text, citations = pack_context_with_citations(top_docs)

        # 5) Add naive KG hints (optional but cheap)
        kg_context = query_knowledge_graph(query)
        kg_info = "\n".join([f"{s} -[{r}]-> {o}" for s, r, o in kg_context]) or "No KG hints."

        # 6) Ask GPT with strict grounding
        system_prompt = (
            "You are a precise AI tutor. Use ONLY the provided context. "
            "If the context is insufficient, say you don't know and suggest what to check next. "
            "Cite evidence using bracketed numbers like [1], [2] that map to the context snippets."
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"=== Context Snippets (cite by [#]) ===\n{context_text}\n\n"
            f"=== Knowledge Graph Hints ===\n{kg_info}\n\n"
            "Write a concise answer (≤ 6 sentences) and include citations like [1], [2] where used."
        )

        answer = call_gpt(system_prompt, user_prompt)

        # 7) TTS
        mp3_path, wav_path = generate_tts(answer)

        return {
            "answer": answer,
            "citations": citations,
            "mp3_path": mp3_path,
            "wav_path": wav_path,
        }, mp3_path

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, None


# -----------------------------
# ROUTES
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global document_text
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(content={"error": "Upload a PDF file"}, status_code=400)

    await reset()

    document_text = extract_text_from_pdf(file)
    if document_text.startswith("❌") or document_text.startswith("⚠️"):
        return JSONResponse(content={"error": document_text}, status_code=500)

    build_knowledge_graph(document_text)
    initialize_retriever(document_text)
    return {"message": "✅ PDF processed, KG built, retriever ready", "text": document_text}


@app.post("/ask_text")
async def ask_text(data: dict):
    question = (data.get("question") or "").strip()
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    response, mp3_path = chat_with_bot(question)
    if "error" in response:
        return JSONResponse(content=response, status_code=500)

    return {
        "answer": response["answer"],
        "citations": response.get("citations", []),
        "audio_url": f"/tts/{os.path.basename(mp3_path)}",
    }


@app.post("/ask_audio")
async def ask_audio(file: UploadFile = File(...)):
    try:
        file_info = {"filename": file.filename, "content_type": file.content_type}
        print("📥 Received audio file:", file_info)

        question = transcribe_audio(file)
        if question.startswith("❌"):
            raise ValueError(f"Transcription failed: {question}")

        response, mp3_path = chat_with_bot(question)
        if "error" in response:
            raise ValueError(f"Bot returned error: {response['error']}")

        return {
            "question": question,
            "answer": response["answer"],
            "citations": response.get("citations", []),
            "audio_url": f"/tts/{os.path.basename(mp3_path)}",
        }

    except Exception as e:
        print("❌ Exception during /ask_audio:")
        traceback.print_exc()
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "filename": file.filename if file else None,
                "content_type": file.content_type if file else None,
            },
            status_code=500,
        )


@app.get("/tts/{filename}")
async def get_tts(filename: str):
    path = os.path.join("static", filename)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Audio not found"}, status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/audio/cloned.wav")
async def get_cloned_audio():
    path = os.path.join("saved_outputs", "latest_output.wav")
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav", filename="cloned.wav")
    return JSONResponse({"status": "error", "message": "Audio not found"}, status_code=404)


@app.post("/reset")
async def reset():
    global retriever, knowledge_graph, graph_triples, document_text
    retriever = None
    knowledge_graph = None
    graph_triples = []
    document_text = ""
    return {"message": "Session reset."}


@app.get("/health")
def health():
    return {"status": "ok"}
