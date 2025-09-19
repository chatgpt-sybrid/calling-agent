import speech_recognition as sr
import pyttsx3
import threading
import queue
import os
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
llm = OllamaLLM(model="llama3.2", temperature=0.3)

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
    Canvi's next reply:
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
speaker.setProperty('rate', 110) 
speaker.setProperty('voice', 'english_us') 
speaker.setProperty('gender', 'female') 

def speak_response():
    while True:
        text = speaker_queue.get()
        if text is None: 
            break
        print("Canvi: Speaking...")
        speaker.say(text)
        speaker.runAndWait()
        print("Canvi: Finished speaking.")
        speaker_queue.task_done()

speaker_thread = threading.Thread(target=speak_response, daemon=True)
speaker_thread.start()

def recognize_speech_and_respond():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Canvi: Listening...")
        try:
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            client_reply = recognizer.recognize_google(audio)
            print(f"Client (Voice): {client_reply}")
            return client_reply
        except sr.UnknownValueError:
            print("Canvi: Could not understand audio, please try again.")
            return ""
        except sr.RequestError as e:
            print(f"Canvi: Speech recognition service error: {e}")
            return ""
        except Exception as e:
            print(f"Canvi: An unexpected error occurred: {e}")
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
    print("\nVoice Cold Call Started (say 'bye' to quit)\n")

    # Initial greeting
    initial_greeting = f"Hello, this is Canvi from Canvas Digital. Hi {client_info['Name']}, how are you doing today?"
    speaker_queue.put(initial_greeting)
    conversation_history += f"Canvi: {initial_greeting}\n"
    
    while True:
        client_reply = recognize_speech_and_respond()

        if client_reply == "NO_RESPONSE":
            no_response_count += 1
            if no_response_count >= max_no_response:
                response_message = "Sorry, are you there? I can't hear you. I'll end the call now. Goodbye!"
                speaker_queue.put(response_message)
                print(f"Canvi: {response_message}")
                break
            else:                
                speaker_queue.put("Sorry, are you there? Can you hear me?") 
                print("Canvi: Sorry, are you there? Can you hear me?")
                continue 
        else:
            no_response_count = 0

        if client_reply.lower() == "bye":
            speaker_queue.put("Thank you for your time. Goodbye!")
            break

        conversation_history += f"Client: {client_reply}\n"
        
        response = llm_stage_reply(client_info, conversation_history, new_offer_details).strip()
        
        if "GOODBYE_CALL" in response:
            speaker_queue.put("Thank you for your time. Goodbye!")
            break
        
        speaker_queue.put(response)
        conversation_history += f"Canvi: {response}\n"

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
    print("\nText Cold Call Started (type 'bye' to quit)\n")

    # Initial greeting
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
    print("Canvas Digital Cold Call System")
    print("=" * 40)
    
    query = input("Type the client name to call: ")
    client_meta = find_client(query)
    
    if not client_meta:
        print(" Client not found in records. Please try another name.")
        return
    
    print("\nSelect call mode:")
    print("1. Voice Call (with speech)")
    print("2.  Text Call (typing)")
    
    choice = input("Choose option (1 or 2): ").strip()

    new_offer_details = input("Enter details for the new opportunity: ") 
    
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