import os
import json
import time
import shutil
import traceback
from tempfile import NamedTemporaryFile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import fitz  # from PyMuPDF
import networkx as nx
import spacy
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from io import BytesIO
from elevenlabs.client import ElevenLabs

# === ENVIRONMENT SETUP ===
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="RagyVerse API", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Eleven Labs SETUP ===
ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
if not ELEVEN_API_KEY or not VOICE_ID:
    raise RuntimeError("Missing ELEVEN_LABS_API_KEY or VOICE_ID")

client_11labs = ElevenLabs(api_key=ELEVEN_API_KEY)

# === LangChain SETUP ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = None
conv_chain = None
document_text = ""
nlp = spacy.load("en_core_web_sm")
knowledge_graph = None
graph_triples = []

ocr_pipe = pipeline("image-text-to-text", model="ds4sd/SmolDocling-256M-preview")


# === UTILITY FUNCTIONS ===

def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        temp_path = NamedTemporaryFile(delete=False, suffix=".pdf").name
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        doc = fitz.open(temp_path)
        extracted_text = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                extracted_text.append(text)
            else:
                for img in page.get_images(full=True):
                    xref = img[0]
                    image = doc.extract_image(xref)
                    image_bytes = image["image"]
                    messages = [{"role": "user", "content": image_bytes}]
                    result = ocr_pipe(messages)
                    if result:
                        extracted_text.append(result[0]["generated_text"])

        os.unlink(temp_path)
        return "\n".join(extracted_text) if extracted_text else "⚠️ No extractable text found"
    except Exception as e:
        return f"❌ Error processing PDF: {e}"


def build_knowledge_graph(text: str):
    global knowledge_graph, graph_triples
    knowledge_graph = nx.DiGraph()
    graph_triples = []

    doc = nlp(text)
    for sent in doc.sents:
        tokens = [token.text for token in sent if not token.is_stop]
        if len(tokens) >= 3:
            subj, rel, obj = tokens[0], "related_to", tokens[-1]
            knowledge_graph.add_edge(subj, obj, relation=rel)
            graph_triples.append((subj, rel, obj))
    return graph_triples


def query_knowledge_graph(query: str):
    if not knowledge_graph:
        return []
    return [
        (u, data['relation'], v)
        for u, v, data in knowledge_graph.edges(data=True)
        if query.lower() in u.lower() or query.lower() in v.lower()
    ]


def initialize_conversational_chain(text: str):
    global conv_chain, retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    prompt = PromptTemplate(
        template="""You are a helpful AI tutor.
Context: {context}
Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )

    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )


def chat_with_bot(query: str):
    global document_text
    if not document_text:
        return {"error": "❌ Upload a PDF first"}, None
    if not conv_chain:
        initialize_conversational_chain(document_text)

    try:
        kg_context = query_knowledge_graph(query)
        kg_info = "\n".join([f"{s} -[{r}]-> {o}" for s, r, o in kg_context]) or "No KG context found"
        enriched = f"{query}\nUse this KG context:\n{kg_info}"

        response = conv_chain.invoke({"question": enriched})
        answer = response.get("answer", "No answer returned.")
        mp3_path, wav_path = generate_tts(answer)
        return {"answer": answer, "mp3_path": mp3_path, "wav_path": wav_path}, mp3_path
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, None


def generate_tts(text: str) -> tuple:
    audio_bytes = b"".join(client_11labs.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    ))

    os.makedirs("static", exist_ok=True)

    filename_base = f"response_{int(time.time())}"
    mp3_path = os.path.join("static", filename_base + ".mp3")
    wav_path = os.path.join("saved_outputs", "latest_output.wav")

    with open(mp3_path, "wb") as f:
        f.write(audio_bytes)

    # Convert to WAV for Unity
    sound = AudioSegment.from_file(mp3_path, format="mp3")
    os.makedirs("saved_outputs", exist_ok=True)
    sound.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

    return mp3_path, wav_path


def transcribe_audio(file: UploadFile) -> str:
    try:
        # Read and buffer audio file
        audio_data = BytesIO(file.file.read())

        # Transcribe using ElevenLabs Speech-to-Text
        transcription = client_11labs.speech_to_text.convert(
            file=audio_data,
            model_id="scribe_v1",
            tag_audio_events=True,
            language_code="eng",
            diarize=True
        )

        return transcription.text
    except Exception as e:
        return f"❌ Error transcribing: {e}"


# === ROUTES ===

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global document_text
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Upload a PDF file"}, status_code=400)

    await reset()

    document_text = extract_text_from_pdf(file)
    if document_text.startswith("❌") or document_text.startswith("⚠️"):
        return JSONResponse(content={"error": document_text}, status_code=500)

    build_knowledge_graph(document_text)
    return {"message": "✅ PDF processed and KG built", "text": document_text}


@app.post("/ask_text")
async def ask_text(data: dict):
    question = data.get("question", "")
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)

    response, mp3_path = chat_with_bot(question)
    if "error" in response:
        return JSONResponse(content=response, status_code=500)

    return {
        "answer": response["answer"],
        "audio_url": f"/tts/{os.path.basename(mp3_path)}"
    }


@app.post("/ask_audio")
async def ask_audio(file: UploadFile = File(...)):
    try:
        # Log file info for debugging
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type
        }
        print("📥 Received audio file:", file_info)

        # Run transcription
        question = transcribe_audio(file)
        if question.startswith("❌"):
            raise ValueError(f"Transcription failed: {question}")

        # Generate response
        response, mp3_path = chat_with_bot(question)
        if "error" in response:
            raise ValueError(f"Bot returned error: {response['error']}")

        # Return final output
        return {
            "question": question,
            "answer": response["answer"],
            "audio_url": f"/tts/{os.path.basename(mp3_path)}"
        }

    except Exception as e:
        print("❌ Exception during /ask_audio:")
        traceback.print_exc()  # Logs the full traceback to the console

        # Return structured error to frontend
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "filename": file.filename if file else None,
                "content_type": file.content_type if file else None
            },
            status_code=500
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
    else:
        return JSONResponse({"status": "error", "message": "Audio not found"}, status_code=404)


@app.post("/reset")
async def reset():
    global memory, conv_chain, retriever, knowledge_graph, graph_triples, document_text
    memory.clear()
    conv_chain = None
    retriever = None
    knowledge_graph = None
    graph_triples = []
    document_text = ""
    return {"message": "Session reset."}

@app.get("/health")
def health():
    return {"status": "ok"}
