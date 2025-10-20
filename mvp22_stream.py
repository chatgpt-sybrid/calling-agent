import sys
import numpy as np
import sounddevice as sd
import queue
import threading
from scipy.io import wavfile
import tempfile
import os

from loguru import logger
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# For STT - using faster-whisper (local)
from faster_whisper import WhisperModel

# For TTS - using kokoro-onnx (local)
import kokoro

# --- RAG/Embedding/LLM setup ---
db_path = r"chroma_db"
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)

def find_client(name_query: str):
    docs = vectorstore.similarity_search(name_query, k=1, filter={"Type": "Client"})
    if not docs:
        return None
    return docs[0].metadata

llm = OllamaLLM(model="llama3.2", temperature=0.1)

full_prompt = PromptTemplate(
    template="""
    You are CANVI, a professional and empathetic sales representative from Canvas Digital. You are on a cold call with a client.

    Your goal is to guide the conversation through four stages:
    1.  **INTRODUCTION**: Greet the client politely and introduce yourself.
    2.  **CONFIRMATION**: Briefly and politely confirm their last purchased service and date. Acknowledge their response, whether they remember or not.
    3.  **ENGAGEMENT**: Present an opportunity for future engagement, inquire about their satisfaction with past services, and understand their current needs or interest in future collaborations. If the client mentions a problem or expresses interest, respond with reassuring and solution-oriented language, aiming to schedule a follow-up meeting. Handle objections politely.
    4.  **CLOSING**: Only if the client explicitly states they are not interested in any future engagement or further discussion, then thank the client for their time and end the call gracefully. If the conversation has reached a natural conclusion where no further action is possible from your end, your response should end with the phrase "GOODBYE_CALL". Otherwise, continue the engagement.

    **CRITICAL RESPONSE RULES:**
    -   Keep responses SHORT - maximum 1-2 sentences only
    -   NEVER give long explanations or multiple points in one response
    -   Ask ONE question at a time
    -   Speak naturally like in a real phone conversation
    -   Be conversational, not formal or wordy
    -   Always maintain a professional yet friendly and empathetic tone
    -   Acknowledge the client's feelings briefly
    -   Do not repeat yourself
    -   Move through the stages logically but naturally

    **Client Data:**
    -   Name: {client_name}
    -   Last Service: {last_service}
    -   Purchase Date: {purchase_date}
    -   New Opportunity: {new_offer_details}

    **Conversation History:**
    {chat_history}
    
    Generate a SHORT response of maximum 1-2 sentences (do not include "Canvi:" prefix):
    
    """,
    input_variables=[
        "client_name",
        "last_service",
        "purchase_date",
        "new_offer_details",
        "chat_history",
    ],
)

def llm_stage_reply(client_info, chat_history, new_offer_details):
    inputs = {
        "client_name": client_info["Name"],
        "last_service": client_info["LastService"],
        "purchase_date": client_info["PurchaseDate"],
        "new_offer_details": new_offer_details,
        "chat_history": chat_history,
    }
    return llm.invoke(full_prompt.format(**inputs))

# --- Audio setup ---
logger.remove(0)
logger.add(sys.stderr, level="INFO")

# Initialize STT model (faster-whisper)
print("Loading STT model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")
print("STT model loaded!")

# Initialize TTS model (kokoro)
print("Loading TTS model...")
tts_pipeline = kokoro.Pipeline(lang_code='a')  # 'a' for American English
print("TTS model loaded!")

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Adjust based on your microphone
SILENCE_DURATION = 1.5  # seconds of silence to stop recording

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.frames = []
        
    def callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def record_until_silence(self):
        """Record audio until silence is detected"""
        self.frames = []
        self.is_recording = True
        silence_frames = 0
        max_silence_frames = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                           callback=self.callback, blocksize=1024):
            logger.info("🎤 Listening... (speak now)")
            
            while True:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    self.frames.append(data)
                    
                    # Check for silence
                    if np.max(np.abs(data)) < SILENCE_THRESHOLD:
                        silence_frames += 1
                        if silence_frames > max_silence_frames:
                            logger.info("Silence detected, processing...")
                            break
                    else:
                        silence_frames = 0
                        
                except queue.Empty:
                    continue
        
        self.is_recording = False
        
        if not self.frames:
            return None
        
        # Combine all frames
        audio_data = np.concatenate(self.frames, axis=0)
        return audio_data

def transcribe_audio(audio_data):
    """Transcribe audio using faster-whisper"""
    if audio_data is None or len(audio_data) == 0:
        return ""
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        wavfile.write(tmp_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    
    try:
        segments, info = stt_model.transcribe(tmp_path, language="en", beam_size=5)
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()
    finally:
        os.unlink(tmp_path)

def speak_text(text):
    """Convert text to speech and play it"""
    logger.info(f"🔊 Speaking: {text}")
    
    # Generate audio using kokoro
    audio_data, sample_rate = tts_pipeline(text)
    
    # Play audio
    sd.play(audio_data, sample_rate)
    sd.wait()  # Wait until audio finishes playing

# --- Main conversation loop ---
def run_conversation(client_info, new_offer_details):
    conversation_history = ""
    recorder = AudioRecorder()
    
    # Start with introduction
    intro = f"Hello, this is Canvi from Canvas Digital. May I speak with {client_info['Name']}?"
    speak_text(intro)
    conversation_history += f"Canvi: {intro}\n"
    
    print("\n📞 Call started. Press Ctrl+C to end.\n")
    
    try:
        while True:
            # Record user speech
            audio_data = recorder.record_until_silence()
            
            # Transcribe
            transcript = transcribe_audio(audio_data)
            
            if not transcript:
                response_text = "I didn't catch that. Could you please repeat?"
                logger.warning("Empty transcript")
            else:
                logger.info(f"Client said: {transcript}")
                conversation_history += f"Client: {transcript}\n"
                
                # Check for goodbye
                if "GOODBYE_CALL" in conversation_history or \
                   any(word in transcript.lower() for word in ["goodbye", "bye", "hang up", "end call"]):
                    response_text = "Thank you for your time. Have a great day!"
                    speak_text(response_text)
                    break
                
                # Generate response
                try:
                    response = llm_stage_reply(client_info, conversation_history, new_offer_details)
                    response_text = response.strip()
                    if response_text.startswith("Canvi:"):
                        response_text = response_text[6:].strip()
                    
                    conversation_history += f"Canvi: {response_text}\n"
                    
                    # Check if agent wants to end call
                    if "GOODBYE_CALL" in response_text:
                        response_text = response_text.replace("GOODBYE_CALL", "").strip()
                        speak_text(response_text)
                        break
                        
                except Exception as e:
                    logger.exception("LLM error")
                    response_text = "Sorry, I had a technical issue. Could we try that again?"
            
            # Speak response
            speak_text(response_text)
            
    except KeyboardInterrupt:
        print("\n\n📞 Call ended by user.")
    except Exception as e:
        logger.exception("Error during call")
    
    print("\n" + "="*50)
    print("CONVERSATION SUMMARY")
    print("="*50)
    print(conversation_history)
    print("="*50)

def main():
    print("Canvas Digital Sales Agent (Local Version)")
    print("=" * 50)
    
    query = input("Enter client name: ")
    client_meta = find_client(query)
    
    if not client_meta:
        print("❌ Client not found in records. Please try another name.")
        return
    
    print(f"\n✅ Client found: {client_meta['Name']}")
    print(f"   Last Service: {client_meta['LastService']}")
    print(f"   Purchase Date: {client_meta['PurchaseDate']}")
    
    new_offer_details = input("\nEnter new opportunity details: ")
    
    print("\n🎙️  Starting voice call...\n")
    
    run_conversation(client_meta, new_offer_details)
    
    print("\nThank you for using Canvas Digital Sales Agent.")

if __name__ == "__main__":
    main()