import speech_recognition as sr
from gtts import gTTS
import pygame
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

# LLM setup
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
    """Invokes the LLM with the full conversation history."""
    inputs = {
        "client_name": client_info["Name"],
        "last_service": client_info["LastService"],
        "purchase_date": client_info["PurchaseDate"],
        "new_offer_details": new_offer_details,
        "chat_history": chat_history,
    }
    
    return llm.invoke(full_prompt.format(**inputs))

# Initialize speech components
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Initialize pygame for audio playback
pygame.mixer.init()

# Test the speaker
print("Testing speaker...")
test_tts = gTTS(text="Speaker test successful", lang='en', slow=False)
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
    test_file = fp.name
    test_tts.save(test_file)
pygame.mixer.music.load(test_file)
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
pygame.mixer.music.unload()
os.unlink(test_file)
print("Speaker is working")

# Initialize Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully") 

def speak(text):
    """TTS using gTTS"""
    print(f"Speaking: {text}")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        os.unlink(temp_file)
        
        print("Finished speaking")
    except Exception as e:
        print(f"Speech error: {e}")

def recognize_speech_whisper():
    """Enhanced speech recognition using Whisper for better accuracy"""
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_file.write(audio.get_wav_data())
                tmp_file.close()
                
                time.sleep(0.1)
                
                print("Processing speech with Whisper...")
                result = whisper_model.transcribe(tmp_file.name)
                client_reply = result["text"].strip()
                
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
                
                if client_reply:
                    print(f"Client: {client_reply}")
                    return client_reply
                else:
                    return ""
                        
            except Exception as whisper_error:
                print(f"Whisper failed, falling back to Google: {whisper_error}")
                client_reply = recognizer.recognize_google(audio)
                if client_reply:
                    print(f"Client: {client_reply}")
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
        speak("Client not found in records. Please try another name.")
        return
    
    no_response_count = 0
    max_no_response = 1

    client_info = {
        "Name": client_meta.get("Name", "Unknown"),
        "LastService": client_meta.get("LastService", "Unknown"),
        "PurchaseDate": client_meta.get("PurchaseDate", "Unknown"),
    }

    conversation_history = ""
    print(f"\nCalling {client_info['Name']}...")
    print("=" * 50)

    initial_greeting = f"Hello, this is Canvi from Canvas Digital. Hi {client_info['Name']}, how are you doing today?"
    print(f"Canvi: {initial_greeting}")
    speak(initial_greeting)
    conversation_history += f"Canvi: {initial_greeting}\n"
    
    while True:
        client_reply = recognize_speech_whisper()

        if client_reply == "NO_RESPONSE":
            no_response_count += 1
            if no_response_count >= max_no_response:
                response_message = "Sorry, are you there? I can't hear you. I'll end the call now. Goodbye!"
                print(f"Canvi: {response_message}")
                speak(response_message)
                break
            else:                
                retry_message = "Sorry, are you there? Can you hear me?"
                print(f"Canvi: {retry_message}")
                speak(retry_message)
                continue 
        else:
            no_response_count = 0

        if client_reply.lower().strip() == "bye":
            goodbye_message = "Thank you for your time. Goodbye!"
            print(f"Canvi: {goodbye_message}")
            speak(goodbye_message)
            break

        conversation_history += f"Client: {client_reply}\n"
        
        try:
            response = llm_stage_reply(client_info, conversation_history, new_offer_details)
            response = response.strip()
            
            if response.startswith("Canvi:"):
                response = response[6:].strip()
            
            if not response or response == "":
                response = "I understand. Let me know if you have any questions about our services."
            
            if "GOODBYE_CALL" in response:
                response = response.replace("GOODBYE_CALL", "").strip()
                
                if response:
                    print(f"Canvi: {response}")
                    speak(response)
                    conversation_history += f"Canvi: {response}\n"
                
                goodbye_message = "Thank you for your time. Goodbye!"
                print(f"Canvi: {goodbye_message}")
                speak(goodbye_message)
                break
            
            print(f"Canvi: {response}")
            speak(response)
            conversation_history += f"Canvi: {response}\n"
            
        except Exception as e:
            print(f"Error: {e}")
            fallback_response = "I apologize, I'm having trouble processing that. Could you please repeat?"
            print(f"Canvi: {fallback_response}")
            speak(fallback_response)
            conversation_history += f"Canvi: {fallback_response}\n"

    print("\nCall ended")
    print("=" * 50)

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
    print(f"\nChat with {client_info['Name']} started")
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
    print("Canvas Digital Sales Agent")
    print("=" * 40)
    
    query = input("Enter client name: ")
    client_meta = find_client(query)
    
    if not client_meta:
        print("Client not found in records. Please try another name.")
        return
    
    print("\nCall Mode Selection:")
    print("1. Voice Call (Real-time conversation)")
    print("2. Text Call (Chat simulation)")
    
    choice = input("Choose option (1 or 2): ").strip()

    new_offer_details = input("Enter new opportunity details: ") 
    
    if choice == "1":
        try:
            cold_call_voice(client_meta, new_offer_details)
        except Exception as e:
            print(f"Voice call error: {e}")
            print("Falling back to text mode...")
            cold_call_text(client_meta, new_offer_details)
    else:
        cold_call_text(client_meta, new_offer_details)

if __name__ == "__main__":
    main()