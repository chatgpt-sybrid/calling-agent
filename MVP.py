import speech_recognition as sr
import pyttsx3
import threading
import queue
import os
import tempfile
import time
import whisper
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

db_path = r"chroma_db"

embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)

def find_client(name_query: str):
    """Retrieves client data by exact name similarity."""
    docs = vectorstore.similarity_search(name_query, k=1, filter={"Type": "Client"})
    if not docs:
        return None
    return docs[0].metadata

# --- LLM ---
llm = OllamaLLM(model="llama3.2", temperature=0.1)

full_prompt = PromptTemplate(
    template="""
    You are CANVI, a professional and empathetic sales representative from Canvas Digital. You are on a cold call with a client.

    Your goal is to guide the conversation through four stages:
    1.  **INTRODUCTION**: Greet the client politely and introduce yourself.
    2.  **CONFIRMATION**: Briefly and politely confirm their last purchased service and date. Acknowledge their response, whether they remember or not.
    3.  **ENGAGEMENT**: Present an opportunity for future engagement, inquire about their satisfaction with past services, and understand their current needs or interest in future collaborations. If the client mentions a problem or expresses interest, respond with reassuring and solution-oriented language, aiming to schedule a follow-up meeting. Handle objections politely, and be precise not give 3-4 line sentence keep it short and concise.
    4.  **CLOSING**: Only if the client explicitly states they are not interested in any future engagement or further discussion, then thank the client for their time and end the call gracefully. If the conversation has reached a natural conclusion where no further action is possible from your end, your response should end with the phrase "GOODBYE_CALL". Otherwise, continue the engagement.

    **Conversation Rules:**
    -   **Always** maintain a professional yet friendly and empathetic tone.
    -   Acknowledge the client's feelings (e.g., if they say they're having a bad day, respond kindly).
    -   Be concise and direct. Do not repeat yourself.
    -   Move through the stages logically. Do not rush the conversation.
    -   Your responses should sound like a natural human conversation.

    **Client Data:**
    -   Name: {client_name}
    -   Last Service: {last_service}
    -   Purchase Date: {purchase_date}
    -   New Opportunity: {new_offer_details}

    **Conversation History:**
    {chat_history}
    
    Generate your response (do not include "Canvi:" prefix):
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
    """Invokes the LLM with the full conversation history."""
    inputs = {
        "client_name": client_info["Name"],
        "last_service": client_info["LastService"],
        "purchase_date": client_info["PurchaseDate"],
        "new_offer_details": new_offer_details,
        "chat_history": chat_history,
    }
    return llm.invoke(full_prompt.format(**inputs))

recognizer = sr.Recognizer()
microphone = sr.Microphone()
speaker = pyttsx3.init()
speaker_queue = queue.Queue() 
speaker.setProperty('rate', 200) 
speaker.setProperty('voice', 'english_us') 
speaker.setProperty('gender', 'female')

# Test the speaker
print("🔊 Testing speaker...")
speaker.say("Speaker test successful")
speaker.runAndWait()
print("✅ Speaker is working!")

# Initialize Whisper model (using base model for good balance of speed/accuracy)
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!") 

def speak_response():
    while True:
        text = speaker_queue.get()
        if text is None: 
            break
        print(f"🔊 Speaking: {text}")
        speaker.say(text)
        speaker.runAndWait()
        print("🔇 Finished speaking")
        speaker_queue.task_done()

speaker_thread = threading.Thread(target=speak_response, daemon=True)
speaker_thread.start()

# Give the speaker thread a moment to start
import time
time.sleep(0.5)
print("✅ Speaker thread started and ready")

def recognize_speech_whisper():
    """Enhanced speech recognition using Whisper for better accuracy"""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎤 Listening...")
            
            # Record audio
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            
            # Try Whisper first (more accurate)
            try:
                # Save audio to temporary file for Whisper
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_file.write(audio.get_wav_data())
                tmp_file.close()
                
                # Small delay to ensure file is fully written
                time.sleep(0.1)
                
                # Use Whisper to transcribe
                print("🧠 Processing speech with Whisper...")
                result = whisper_model.transcribe(tmp_file.name)
                client_reply = result["text"].strip()
                
                # Clean up temp file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass  # Ignore cleanup errors
                
                if client_reply:
                    print(f"👤 Client: {client_reply}")
                    return client_reply
                else:
                    return ""
                        
            except Exception as whisper_error:
                print(f"Whisper failed, falling back to Google: {whisper_error}")
                # Fallback to Google speech recognition
                client_reply = recognizer.recognize_google(audio)
                if client_reply:
                    print(f"👤 Client: {client_reply}")
                    return client_reply
                else:
                    return ""
                    
    except sr.WaitTimeoutError:
        return "NO_RESPONSE"
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return "NO_RESPONSE"

def cold_call_voice(client_meta, new_offer_details):
    """Simulates a cold call conversation with a client using voice."""
    if not client_meta:
        speaker_queue.put("Client not found in records. Please try another name.")
        return
    
    no_response_count = 0 # Initialize counter for no responses
    max_no_response = 1   # Maximum allowed no responses before ending call (1 for 30 seconds timeout)

    client_info = {
        "Name": client_meta.get("Name", "Unknown"),
        "LastService": client_meta.get("LastService", "Unknown"),
        "PurchaseDate": client_meta.get("PurchaseDate", "Unknown"),
    }

    conversation_history = ""
    print(f"\n Calling {client_info['Name']}...")
    print("=" * 50)

    # Initial greeting
    initial_greeting = f"Hello, this is Canvi from Canvas Digital. Hi {client_info['Name']}, how are you doing today?"
    print(f"🔊 Adding initial greeting to speaker queue: {initial_greeting}")
    speaker_queue.put(initial_greeting)
    conversation_history += f"Canvi: {initial_greeting}\n"
    
    while True:
        client_reply = recognize_speech_whisper()

        if client_reply == "NO_RESPONSE":
            no_response_count += 1
            if no_response_count >= max_no_response:
                response_message = "Sorry, are you there? I can't hear you. I'll end the call now. Goodbye!"
                speaker_queue.put(response_message)
                break
            else:                
                speaker_queue.put("Sorry, are you there? Can you hear me?") 
                continue 
        else:
            no_response_count = 0

        if client_reply.lower() == "bye":
            speaker_queue.put("Thank you for your time. Goodbye!")
            break

        conversation_history += f"Client: {client_reply}\n"
        
        try:
            response = llm_stage_reply(client_info, conversation_history, new_offer_details)
            response = response.strip()
            
            # Remove "Canvi:" prefix if it exists
            if response.startswith("Canvi:"):
                response = response[6:].strip()
            
            if not response or response == "":
                response = "I understand. Let me know if you have any questions about our services."
            
            if "GOODBYE_CALL" in response:
                speaker_queue.put("Thank you for your time. Goodbye!")
                break
            
            # Speak the response
            print(f"🤖 Canvi: {response}")
            print(f"🔊 Adding to speaker queue: {response}")
            speaker_queue.put(response)
            conversation_history += f"Canvi: {response}\n"
            
        except Exception as e:
            fallback_response = "I apologize, I'm having trouble processing that. Could you please repeat?"
            speaker_queue.put(fallback_response)
            conversation_history += f"Canvi: {fallback_response}\n"

    speaker_queue.put(None) 
    speaker_thread.join()

def cold_call_text(client_meta, new_offer_details):
    """Simulates a cold call conversation with a client."""
    if not client_meta:
        print("Canvi: Client not found in records. Please try another name.")
        return

    client_info = {
        "Name": client_meta.get("Name", "Unknown"),
        "LastService": client_meta.get("LastService", "Unknown"),
        "PurchaseDate": client_meta.get("PurchaseDate", "Unknown"),
    }

    conversation_history = ""
    print(f"\n Chat with {client_info['Name']} started")
    print("=" * 50)

    initial_greeting = f"Hello, this is Canvi from Canvas Digital. Hi {client_info['Name']}, how are you doing today?"
    print(f"Canvi: {initial_greeting}")
    conversation_history += f"Canvi: {initial_greeting}\n"
    
    while True:
        client_reply = input("Client: ")
        if client_reply.lower() == "bye":
            print("Canvi: Thank you for your time. Goodbye!")
            break

        conversation_history += f"Client: {client_reply}\n"
        
        response = llm_stage_reply(client_info, conversation_history, new_offer_details).strip()
        
        if "GOODBYE_CALL" in response:
            print("Canvi: Thank you for your time. Goodbye!")
            break
        
        print(f"Canvi: {response}")
        conversation_history += f"Canvi: {response}\n"


def main():
    print(" Canvas Digita Sales Agent")
    print("=" * 40)
    
    query = input("👤 Enter client name: ")
    client_meta = find_client(query)
    
    if not client_meta:
        print(" Client not found in records. Please try another name.")
        return
    
    print("\n Call Mode Selection:")
    print("1. Voice Call (Real-time conversation)")
    print("2. Text Call (Chat simulation)")
    
    choice = input("Choose option (1 or 2): ").strip()

    new_offer_details = input(" Enter new opportunity details: ") 
    
    if choice == "1":
        try:
            cold_call_voice(client_meta, new_offer_details)
        except Exception as e:
            print(f" Voice call error: {e}")
            print(" Falling back to text mode...")
            cold_call_text(client_meta, new_offer_details)
    else:
        cold_call_text(client_meta, new_offer_details)

if __name__ == "__main__":
    main()