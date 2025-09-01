# main.py
import os
import json
import re
import io
import wave
import time
import traceback
import threading
from uuid import uuid4
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Optional, List, Tuple

import fitz  # PyMuPDF
import networkx as nx
from fastapi import FastAPI, File, UploadFile, Request
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
    allow_origins=["*"],  # tighten in prod
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
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-nano")  # fastest for chat; change if you want

ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
if not ELEVEN_API_KEY or not VOICE_ID:
    raise RuntimeError("Missing ELEVEN_LABS_API_KEY or VOICE_ID")
client_11labs = ElevenLabs(api_key=ELEVEN_API_KEY)

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", "eleven_multilingual_v2")
# Allowed: mp3_22050_32, mp3_44100_32|64|96|128|192, opus_48000_* (avoid for Unity), pcm_*
TTS_FORMAT = os.getenv("TTS_MP3_FORMAT", "mp3_22050_32").lower()
# clip length for SSE chunking (in ms)
TTS_CLIP_MS = int(os.getenv("TTS_CLIP_MS", "1600"))

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

# ---------------------------------
# Utils
# ---------------------------------
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

def _mime_for_ext(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".mp3"): return "audio/mpeg"
    if lower.endswith(".wav"): return "audio/wav"
    if lower.endswith(".opus"): return "audio/ogg"
    return "application/octet-stream"

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
        snippet = (d.page_content or "").strip()[:500]
        cid = d.metadata.get("chunk_id", idx)
        snippets.append(f"[{idx}] {snippet}")
        cites.append({"id": idx, "chunk_id": cid})
    return "\n".join(snippets), cites

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

        return {"answer": answer, "citations": citations}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        end_total()

# ---------------------------------
# MP3 frame utilities (cut valid mini-MP3 files while streaming)
# ---------------------------------
_MP3_BITRATES = {
    "MPEG1": { 0:0, 1:32, 2:40, 3:48, 4:56, 5:64, 6:80, 7:96,
               8:112, 9:128, 10:160, 11:192, 12:224, 13:256, 14:320, 15:0 },
    "MPEG2": { 0:0, 1:8, 2:16, 3:24, 4:32, 5:40, 6:48, 7:56,
               8:64, 9:80, 10:96, 11:112, 12:128, 13:144, 14:160, 15:0 }
}
_MP3_SAMPLERATES = {
    "MPEG1":  {0:44100, 1:48000, 2:32000, 3:0},
    "MPEG2":  {0:22050, 1:24000, 2:16000, 3:0},
    "MPEG25": {0:11025, 1:12000, 2:8000,  3:0},
}

def _mp3_frame_info(hdr: bytes):
    """Return (frame_len_bytes, frame_ms, samplerate) or (None, None, None)."""
    if len(hdr) < 4: return (None, None, None)
    b0,b1,b2,b3 = hdr[0],hdr[1],hdr[2],hdr[3]
    # 11-bit sync 0xFFE
    if (b0 != 0xFF) or ((b1 & 0xE0) != 0xE0):
        return (None, None, None)

    version_id = (b1 >> 3) & 0x03  # 00:2.5, 10:2, 11:1
    layer = (b1 >> 1) & 0x03       # 01 = Layer III
    if layer != 1:  # only Layer III
        return (None, None, None)

    bitrate_idx = (b2 >> 4) & 0x0F
    sr_idx = (b2 >> 2) & 0x03
    padding = (b2 >> 1) & 0x01

    if version_id == 3:   ver = "MPEG1"
    elif version_id == 2: ver = "MPEG2"
    elif version_id == 0: ver = "MPEG25"
    else: return (None, None, None)

    sr = _MP3_SAMPLERATES[ver].get(sr_idx, 0)
    if sr == 0: return (None, None, None)

    br_tab = _MP3_BITRATES["MPEG1" if ver=="MPEG1" else "MPEG2"]
    kbps = br_tab.get(bitrate_idx, 0)
    if kbps == 0: return (None, None, None)

    if ver == "MPEG1":
        frame_len = int((144000 * kbps) / sr) + padding
        samples = 1152
    else:
        frame_len = int((72000  * kbps) / sr) + padding
        samples = 576

    frame_ms = (samples * 1000.0) / sr
    return (frame_len, frame_ms, sr)

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
        _ = rr.predict([("warmup", "warmup")])
    except Exception:
        pass
    # TTS warm (credential sanity check)
    try:
        _ = client_11labs.text_to_speech.convert(
            text="Hello.",
            voice_id=VOICE_ID,
            model_id=TTS_MODEL_ID,
            output_format="mp3_22050_32",
        )
        for _chunk in _:
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
    """Legacy single-shot path (kept for compatibility). Use /ask_text_stream for clip-by-clip."""
    question = (data.get("question") or "").strip()
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    response = chat_with_bot_text_only(question)
    if "error" in response:
        return JSONResponse(content=response, status_code=500)

    # Create one full MP3 (blocking) for backward-compatibility (optional).
    # If you prefer not to block here, you can remove this and return audio_url=None.
    text = response["answer"]
    fmt = "mp3_22050_32"
    os.makedirs("static", exist_ok=True)
    ts = int(time.time())
    fname = f"response_{ts}.mp3"
    fpath = os.path.join("static", fname)
    try:
        mp3_bytes = b"".join(
            client_11labs.text_to_speech.convert(
                text=text,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL_ID,
                output_format=fmt,
            )
        )
        _atomic_write_bytes(fpath, mp3_bytes)
        audio_url = f"/tts/{fname}"
    except Exception as e:
        print("⚠️ TTS full make failed:", e)
        audio_url = None

    return {
        "answer": response["answer"],
        "citations": response.get("citations", []),
        "audio_url": audio_url,
    }

@app.post("/ask_text_stream")
async def ask_text_stream(req: Request):
    """
    SSE that:
      - emits {"type":"answer","text": "..."}
      - then repeatedly emits {"type":"clip","url": "/tts/clip_*.mp3","i": n}
      - finally emits {"type":"done"}
    """
    try:
        payload = await req.json()
    except Exception:
        return JSONResponse({"error":"Invalid JSON"}, status_code=400)

    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error":"No question provided"}, status_code=400)

    # Get text answer first (non-blocking TTS)
    resp = chat_with_bot_text_only(question)
    if "error" in resp:
        return JSONResponse(resp, status_code=500)
    answer = resp["answer"]

    # Force MP3 for Unity friendliness
    fmt = TTS_FORMAT if TTS_FORMAT.startswith("mp3") else "mp3_22050_32"
    os.makedirs("static", exist_ok=True)

    def gen():
        # helper to send SSE
        def send(ev: dict):
            return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        # 1) Send the text ASAP
        yield send({"type":"answer","text": answer})

        uid = f"{int(time.time())}_{uuid4().hex[:6]}"
        buf = bytearray()     # unread stream bytes
        stash = bytearray()   # bytes accumulated since last emitted clip
        frames_ms_accum = 0.0
        clip_idx = 0

        try:
            stream = client_11labs.text_to_speech.convert(
                text=answer,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL_ID,
                output_format=fmt,
            )

            for chunk in stream:
                if not chunk:
                    continue
                buf.extend(chunk)

                # Parse complete MP3 frames; append each to stash; emit when we hit ~TTS_CLIP_MS
                i = 0
                n = len(buf)
                consumed = 0
                while i + 4 <= n:
                    fl, fms, _sr = _mp3_frame_info(buf[i:i+4])
                    if fl is None or i + fl > n:
                        i += 1
                        continue
                    # complete frame present
                    frame = buf[i:i+fl]
                    stash.extend(frame)
                    frames_ms_accum += fms
                    i += fl
                    consumed = i

                    if frames_ms_accum >= TTS_CLIP_MS:
                        clip_idx += 1
                        fname = f"clip_{uid}_{clip_idx:02d}.mp3"
                        fpath = os.path.join("static", fname)
                        _atomic_write_bytes(fpath, bytes(stash))
                        stash.clear()
                        frames_ms_accum = 0.0
                        yield send({"type":"clip","url": f"/tts/{fname}","i": clip_idx})

                if consumed > 0:
                    del buf[:consumed]

            # End-of-stream: flush remaining bytes as a final small clip
            if stash:
                clip_idx += 1
                fname = f"clip_{uid}_{clip_idx:02d}.mp3"
                fpath = os.path.join("static", fname)
                _atomic_write_bytes(fpath, bytes(stash))
                yield send({"type":"clip","url": f"/tts/{fname}","i": clip_idx})

        except Exception as e:
            yield send({"type":"error","message": str(e)})

        # 3) Done
        yield send({"type":"done"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    })

@app.get("/tts/{filename}")
async def get_tts(filename: str):
    """Serve finalized audio files (clips or full)."""
    path = os.path.join("static", filename)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Audio not found"}, status_code=404)
    resp = FileResponse(path, media_type=_mime_for_ext(filename))
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# Legacy route kept
@app.get("/audio/cloned.wav")
async def get_cloned_audio():
    # We primarily write mp3 clips in this flow; keep compatibility if you still generate cloned.*
    for name, mt in [("cloned.wav","audio/wav"), ("cloned.mp3","audio/mpeg"), ("cloned.opus","audio/ogg")]:
        p = os.path.join("saved_outputs", name)
        if os.path.exists(p):
            return FileResponse(p, media_type=mt, filename="cloned.wav")
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
