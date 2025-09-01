import os
import json
import time
import traceback
import threading
from uuid import uuid4
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Optional, List, Tuple

import fitz  # PyMuPDF
import networkx as nx
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs
from openai import OpenAI

# LangChain bits
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------------------------
# ENV & APP
# ---------------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _timed(label: str):
    t0 = time.perf_counter()
    def end():
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"⏱ {label}: {dt:.1f} ms")
    return end

app = FastAPI(title="RagyVerse API (GPT-powered)", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature knobs
ENABLE_OCR = os.getenv("ENABLE_OCR", "0") == "1"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))  # tight for speed

# ---------------------------------
# API KEYS / CLIENTS
# ---------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
oai_client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # for speed, gpt-4.1-nano is fastest

ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
if not ELEVEN_API_KEY or not VOICE_ID:
    raise RuntimeError("Missing ELEVEN_LABS_API_KEY or VOICE_ID")
client_11labs = ElevenLabs(api_key=ELEVEN_API_KEY)

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "eleven_multilingual_v2")
# ElevenLabs allowed formats include:
# mp3_22050_32, mp3_44100_32|64|96|128|192, pcm_8000|16000|22050|24000|44100|48000, ulaw_8000, alaw_8000, opus_48000_32|64|96|128|192
TTS_FORMAT = os.getenv("TTS_MP3_FORMAT", "mp3_22050_32")

# We keep MAX_SPOKEN_CHARS for compatibility, but won't use it to truncate.
MAX_SPOKEN_CHARS = int(os.getenv("MAX_SPOKEN_CHARS", "420"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))

# ---------------------------------
# GLOBAL STATE
# ---------------------------------
document_text = ""
knowledge_graph: Optional[nx.DiGraph] = None
graph_triples: List[tuple] = []
retriever = None

# Lazy caches
_spacy_nlp = None
_embeddings = None
_reranker = None
_ocr_pipe = None

# TTS stream registry
_tts_lock = threading.Lock()
# filename -> {"format": str, "ext": str, "started": float}
_tts_registry = {}

# ---------------------------------
# Utils
# ---------------------------------
import io, wave, re

def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw PCM into a RIFF/WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)  # 2 bytes = 16-bit
        w.setframerate(sample_rate)
        # ensure even length for 16-bit samples
        if sampwidth == 2 and (len(pcm_bytes) % 2) != 0:
            pcm_bytes = pcm_bytes[:-1]
        w.writeframes(pcm_bytes)
    return buf.getvalue()

def _parse_rate(fmt: str, default: int = 16000) -> int:
    m = re.search(r'_(\d{4,5})', fmt)
    return int(m.group(1)) if m else default

def _infer_ext(fmt: str) -> str:
    f = fmt.lower()
    if f.startswith("mp3"): return "mp3"
    if f.startswith("opus"): return "opus"
    if f.startswith("pcm"): return "wav"  # we wrap to WAV
    if f.startswith("ulaw") or f.startswith("alaw"): return "wav"  # commonly wrapped to WAV for playback
    return "bin"

def _atomic_write_bytes(target_path: str, data: bytes):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = os.path.join(
        os.path.dirname(target_path),
        f".tmp_{os.path.basename(target_path)}_{int(time.time()*1000)}"
    )
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)

# ---------------------------------
# NLP helpers
# ---------------------------------
def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        kwargs = {}
        if "bge" in model_name.lower():
            kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        _embeddings = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    return _embeddings

def get_ocr():
    global _ocr_pipe
    if _ocr_pipe is None:
        from transformers import pipeline
        model_name = os.getenv("OCR_MODEL", "Salesforce/blip-image-captioning-base")
        _ocr_pipe = pipeline("image-to-text", model=model_name)
    return _ocr_pipe

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ---------------------------------
# GPT helper
# ---------------------------------
def call_gpt(system_prompt: str, user_prompt: str) -> str:
    resp = oai_client.responses.create(
        model=GPT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )
    try:
        return resp.output_text.strip()
    except Exception:
        return (getattr(resp, "output", "") or "").strip() or "No answer returned."

# ---------------------------------
# PDF → Text (optional OCR)
# ---------------------------------
def extract_text_from_pdf(file: UploadFile) -> str:
    end_total = _timed("pdf_total")
    temp_path = None
    doc = None
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
                        pass

        out = "\n".join(extracted).strip() if extracted else "⚠️ No extractable text found"
        return out
    except Exception as e:
        return f"❌ Error processing PDF: {e}"
    finally:
        try:
            if doc: doc.close()
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception: pass
        end_total()

# ---------------------------------
# Simple Knowledge Graph
# ---------------------------------
def build_knowledge_graph(text: str):
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

# ---------------------------------
# Retrieval (FAISS) + MMR + Rerank
# ---------------------------------
def initialize_retriever(text: str):
    global retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents([Document(page_content=text)])

    docs = []
    for i, d in enumerate(chunks):
        d.metadata = {"chunk_id": i}
        docs.append(d)

    vectorstore = FAISS.from_documents(docs, get_embeddings())
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 24, "lambda_mult": 0.5}
    )

def retrieve_pool(query: str) -> List[Document]:
    if not retriever:
        return []
    return retriever.invoke(query)

def rerank_docs(query: str, docs: List[Document], top_k: int = TOP_K_RERANK) -> List[Document]:
    if not docs:
        return []
    rr = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = rr.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

def pack_context_with_citations(docs: List[Document]) -> Tuple[str, List[dict]]:
    snippets = []
    cites: List[dict] = []
    for idx, d in enumerate(docs, 1):
        snippet = (d.page_content or "").strip()[:500]  # lean for fewer tokens
        cid = d.metadata.get("chunk_id", idx)
        snippets.append(f"[{idx}] {snippet}")
        cites.append({"id": idx, "chunk_id": cid})
    return "\n".join(snippets), cites

# ---------------------------------
# TTS: background generation + streaming
# ---------------------------------
def _start_tts_background(answer_text: str, fmt: str) -> str:
    """
    Start ElevenLabs TTS in a background thread.
    Returns the filename (with extension) that the client can GET via /tts/{filename}.
    """
    ext = _infer_ext(fmt)
    base = f"response_{int(time.time())}_{uuid4().hex[:8]}"
    final_name = f"{base}.{ext}"
    final_path = os.path.join("static", final_name)

    # record registry (for visibility / debugging)
    with _tts_lock:
        _tts_registry[final_name] = {"format": fmt, "ext": ext, "started": time.time()}

    def worker():
        try:
            # MP3/OPUS path: stream-write into a temp file (for live tail)
            if ext in ("mp3", "opus"):
                tmp_path = os.path.join("static", f".tmp_{final_name}")
                with open(tmp_path, "wb") as wf:
                    for chunk in client_11labs.text_to_speech.convert(
                        text=answer_text,
                        voice_id=VOICE_ID,
                        model_id=TTS_MODEL_ID,
                        output_format=fmt,
                    ):
                        wf.write(chunk)
                os.replace(tmp_path, final_path)

                # update cloned.* (latest)
                cloned = os.path.join("saved_outputs", f"cloned.{ext}")
                try:
                    with open(final_path, "rb") as rf:
                        _atomic_write_bytes(cloned, rf.read())
                except Exception:
                    pass

            else:
                # PCM/ULAW/ALAW path: accumulate and wrap to WAV, then write final in one go
                pcm_bytes = b"".join(client_11labs.text_to_speech.convert(
                    text=answer_text,
                    voice_id=VOICE_ID,
                    model_id=TTS_MODEL_ID,
                    output_format=fmt,
                ))
                # sample rate from format, default 16k
                sr = _parse_rate(fmt, default=16000)
                wav_bytes = _pcm_to_wav_bytes(pcm_bytes, sample_rate=sr, channels=1, sampwidth=2)
                _atomic_write_bytes(final_path, wav_bytes)

                # update cloned.wav
                cloned = os.path.join("saved_outputs", "cloned.wav")
                _atomic_write_bytes(cloned, wav_bytes)

        except Exception as e:
            print(f"❌ TTS worker error for {final_name}: {e}")
        finally:
            pass

    threading.Thread(target=worker, daemon=True).start()
    return final_name

def _mime_for_ext(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".mp3"): return "audio/mpeg"
    if lower.endswith(".wav"): return "audio/wav"
    if lower.endswith(".opus"): return "audio/ogg"  # Opus-in-Ogg
    return "application/octet-stream"

# ---------------------------------
# STT
# ---------------------------------
def transcribe_audio(file: UploadFile) -> str:
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

# ---------------------------------
# Chat core (returns answer text only; TTS is decoupled)
# ---------------------------------
def chat_with_bot_text_only(query: str):
    end_total = _timed("chat_total")
    try:
        global document_text, retriever
        if not document_text:
            return {"error": "❌ Upload a PDF first"}

        if not retriever:
            initialize_retriever(document_text)

        end_step = _timed("retrieve_pool")
        pool = retrieve_pool(query)
        end_step()

        end_step = _timed("rerank_docs")
        top_docs = rerank_docs(query, pool, top_k=TOP_K_RERANK)
        end_step()

        end_step = _timed("pack_context")
        context_text, citations = pack_context_with_citations(top_docs)
        end_step()

        kg_context = query_knowledge_graph(query)
        kg_info = "\n".join([f"{s} -[{r}]-> {o}" for s, r, o in kg_context]) or "No KG hints."

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

        end_step = _timed("openai_answer")
        answer = call_gpt(system_prompt, user_prompt)
        end_step()

        return {
            "answer": answer,
            "citations": citations,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        end_total()

# ---------------------------------
# ROUTES
# ---------------------------------
@app.on_event("startup")
async def warm_start():
    try:
        get_spacy()
    except Exception:
        pass
    try:
        get_embeddings()
    except Exception:
        pass
    try:
        rr = get_reranker()
        _ = rr.predict([("warmup", "warmup")])  # lightweight warm
    except Exception:
        pass
    # TTS warm (non-blocking/logging only)
    try:
        # Tiny ping to ensure credentials OK (no need to b"".join big audio)
        _ = client_11labs.text_to_speech.convert(
            text="Hello.",
            voice_id=VOICE_ID,
            model_id=TTS_MODEL_ID,
            output_format=TTS_FORMAT,
        )
        # consume just the first tiny chunk if any
        for i, chunk in enumerate(_):
            # don't keep the generator around; this is just a credentials check
            break
        print("✅ TTS warm")
    except Exception as e:
        print(f"⚠️ TTS warm failed: {e}")
    print("✅ Warm start complete")

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

    response = chat_with_bot_text_only(question)
    if "error" in response:
        return JSONResponse(content=response, status_code=500)

    # Start TTS in background immediately; return same-shaped payload as before
    filename = _start_tts_background(response["answer"], fmt=TTS_FORMAT)
    return {
        "answer": response["answer"],
        "citations": response.get("citations", []),
        "audio_url": f"/tts/{filename}",
    }

@app.post("/ask_audio")
async def ask_audio(file: UploadFile = File(...)):
    try:
        file_info = {"filename": file.filename, "content_type": file.content_type}
        print("📥 Received audio file:", file_info)

        question = transcribe_audio(file)
        if question.startswith("❌"):
            raise ValueError(f"Transcription failed: {question}")

        response = chat_with_bot_text_only(question)
        if "error" in response:
            raise ValueError(f"Bot returned error: {response['error']}")

        filename = _start_tts_background(response["answer"], fmt=TTS_FORMAT)
        return {
            "question": question,
            "answer": response["answer"],
            "citations": response.get("citations", []),
            "audio_url": f"/tts/{filename}",
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
    """
    Dual-mode:
      - If the finalized file exists, serve it with correct MIME.
      - Else, if an MP3/OPUS temp file is being written, 'tail' it so the client gets bytes ASAP.
      - Else (PCM path), wait until final WAV appears and then stream it.
    """
    static_dir = "static"
    path = os.path.join(static_dir, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type=_mime_for_ext(filename))

    # If not ready, attempt to stream from temp (MP3/OPUS path) or wait for final (PCM path)
    fmt = TTS_FORMAT.lower()
    ext = _infer_ext(fmt)
    tmp_path = os.path.join(static_dir, f".tmp_{filename}")  # mp3/opus temp pattern

    def tail_file():
        last = 0
        # If PCM: no usable tmp; wait until final exists then stream
        mp3_or_opus = ext in ("mp3", "opus")
        while True:
            # If final file appears while waiting, stream it and exit
            if os.path.exists(path):
                with open(path, "rb") as f:
                    f.seek(last)
                    chunk = f.read(64 * 1024)
                    while chunk:
                        yield chunk
                        last = f.tell()
                        chunk = f.read(64 * 1024)
                break

            # If we have a temp file (mp3/opus), tail it as it's being written
            if mp3_or_opus and os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    f.seek(last)
                    chunk = f.read(64 * 1024)
                    while chunk:
                        yield chunk
                        last = f.tell()
                        chunk = f.read(64 * 1024)

            # brief sleep to avoid busy loop
            time.sleep(0.05)

    # pick a MIME to help the browser start decoding correctly
    mt = _mime_for_ext(filename if ext in ("mp3", "opus") else (filename[:-3] + "wav"))
    return StreamingResponse(tail_file(), media_type=mt)

# keep your existing route name/path the same
@app.get("/audio/cloned.wav")
async def get_cloned_audio():
    # prefer actual WAV if present
    wav_p = os.path.join("saved_outputs", "cloned.wav")
    if os.path.exists(wav_p):
        return FileResponse(
            wav_p, media_type="audio/wav", filename="cloned.wav"
        )
    # fallbacks (serve under .wav name for legacy client)
    mp3_p = os.path.join("saved_outputs", "cloned.mp3")
    if os.path.exists(mp3_p):
        return FileResponse(
            mp3_p, media_type="audio/mpeg", filename="cloned.wav"
        )
    opus_p = os.path.join("saved_outputs", "cloned.opus")
    if os.path.exists(opus_p):
        return FileResponse(
            opus_p, media_type="audio/ogg", filename="cloned.wav"
        )
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
