from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import numpy as np
import faiss
import requests
from typing import Optional
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class TTSRequest(BaseModel):
    id: str
    question: str
    language: str

responses_store = {}
# CORS configuration
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "https://ragbot-inb0.onrender.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"]
)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize PDF and Vector DB
try:
    # Load and split PDF document
    loader = PyPDFLoader("iesc111.pdf")
    pages = loader.load_and_split()
    text = "\n".join([doc.page_content for doc in pages])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    content_list = [doc.page_content for doc in docs]

    # Load FAISS index
    index = faiss.read_index("handbook_index.faiss")
    logger.info("Successfully loaded PDF and FAISS index")
except Exception as e:
    logger.error(f"Error initializing PDF and Vector DB: {str(e)}")
    raise

class Query(BaseModel):
    question: str

def get_embeddings(text):
    """Generate embeddings for the given text using Gemini API"""
    try:
        model = 'models/embedding-001'
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return embedding['embedding']
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def get_relevant_docs(user_query, top_k=3):
    """Retrieve relevant documents from the vector database"""
    try:
        query_embedding = np.array(get_embeddings(user_query)).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        relevant_docs = [content_list[i] for i in indices[0]]
        return relevant_docs
    except Exception as e:
        logger.error(f"Error retrieving relevant documents: {str(e)}")
        raise

def make_rag_prompt(query, relevant_passage):
    """Create a RAG prompt combining the query and relevant passages"""
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"You are a helpful and informative assistant that answers questions using the reference passage below. "
        f"Respond conversationally and make your answers easy to understand. "
        f"If the passage doesn't contain relevant information, acknowledge that and provide a general response.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt

def generate_response(prompt):
    """Generate a response using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-8b')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def get_weather(city):
    """Get weather information for a specified city"""
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.json()
            description = weather_data['weather'][0]['description']
            temp = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            
            weather_response = (
                f"The weather in {city.title()} is currently {description} with a "
                f"temperature of {temp:.1f}°C and humidity of {humidity}%. "
                f"Is there anything specific about the weather you'd like to know?"
            )
            return {"response": weather_response, "type": "weather"}
            
        elif response.status_code == 404:
            return {
                "response": f"I couldn't find weather data for '{city}'. Please check the city name and try again.",
                "type": "error"
            }
        else:
            return {
                "response": f"There was an error fetching weather data (Error {response.status_code}). Please try again later.",
                "type": "error",
                "city": city
            }
            
    except requests.Timeout:
        return {
            "response": "The weather service is taking too long to respond. Please try again later.",
            "type": "error"
        }
    except Exception as e:
        logger.error(f"Error in weather service: {str(e)}")
        return {
            "response": "An error occurred while fetching weather data. Please try again later.",
            "type": "error"
        }

def generate_greeting_response(query):
    """Generate a response to greetings"""
    greeting_prompt = (
        f"Generate a friendly and welcoming response to this greeting: '{query}'\n"
        f"Keep it natural and conversational, and offer to help."
    )
    return generate_response(greeting_prompt)

def generate_answer(query):
    """Generate an answer using RAG"""
    relevant_text = get_relevant_docs(query)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    return generate_response(prompt)

@app.post("/agent")
async def agent_endpoint(query: Query):
    """Main endpoint for handling all types of queries"""
    try:
        question = query.question.lower()
        
        # Classification prompt
        classification_prompt = """
        Analyze the following input and classify it into exactly ONE of these categories:
        1. 'greeting' - if it's a greeting, introduction, or general pleasantry
        2. 'question' - if it's asking for information or clarification
        3. 'weather' - if it's asking about weather conditions
        
        If it's a weather query, also extract the city name.
        
        Input: '{input}'
        
        Respond in this exact format:
        TYPE: [classification]
        CITY: [city name if weather query, otherwise 'none']
        """
        
        # Get classification
        classification_response = generate_response(
            classification_prompt.format(input=question)
        ).lower()
        
        # Parse classification
        response_lines = classification_response.strip().split('\n')
        input_type = response_lines[0].split(':')[1].strip()
        
        # Handle different query types
        if 'greeting' in input_type:
            response = generate_greeting_response(question)
            return {"response": response, "type": "greeting"}
            
        elif 'weather' in input_type:
            city = response_lines[1].split(':')[1].strip()
            if city != 'none':
                return get_weather(city)
            else:
                return {
                    "response": "I noticed you're asking about weather, but I couldn't identify the city. Can you please specify?",
                    "type": "error"
                }
                
        elif 'question' in input_type:
            response = generate_answer(question)
            return {"response": response, "type": "answer"}
            
        else:
            return {
                "response": "I'm not sure how to respond to that. Can you please rephrase?",
                "type": "error"
            }
    
    except Exception as e:
        logger.error(f"Error in agent endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")



import requests
@app.post("/text-to-speech/")
async def text_to_speech(request: Request):
    """Endpoint for text-to-speech conversion"""
    try:
        data = await request.json()
        response_text = data.get('question')
        target_language = data.get('language')
        
        # Log the original response text
        logger.info(f"Original response text: {response_text}")

        # Translate text to the target language
        translation_url = "https://api.sarvam.ai/translate"
        translation_payload = {
            "source_language_code": "en-IN",
            "target_language_code": target_language,
            "input": response_text,
            "model": "mayura:v1"
        }
        translation_headers = {
            "Content-Type": "application/json",
            "api-subscription-key": os.getenv('SARVAAM_API_KEY')  # Include your API key here
        }

        # Log the translation payload
        logger.info(f"Sending translation request with payload: {translation_payload}")

        translation_response = requests.post(translation_url, json=translation_payload, headers=translation_headers)

        # Log the full response from the translation API
        logger.info(f"Translation API response: {translation_response.text}")

        if translation_response.status_code != 200:
            logger.error(f"Translation failed: {translation_response.text}")
            raise HTTPException(status_code=translation_response.status_code, detail="Translation failed")

        translated_text = translation_response.json().get("translated_text")

        # Log the translated text
        logger.info(f"Translated text: {translated_text}")

        if not translated_text:
            raise HTTPException(status_code=500, detail="Translation failed")

        # Convert translated text to speech
        tts_url = "https://api.sarvam.ai/text-to-speech"
        tts_payload = {
            "inputs": [translated_text],
            "target_language_code": target_language,
            "speaker": "arvind",  # Adjust this as needed
            "model": "bulbul:v1"
        }

        # Log the TTS payload
        logger.info(f"Sending TTS request with payload: {tts_payload}")

        tts_response = requests.post(tts_url, json=tts_payload, headers=translation_headers)

        if tts_response.status_code != 200:
            logger.error(f"Text-to-speech conversion failed: {tts_response.text}")
            raise HTTPException(status_code=tts_response.status_code, detail="Text-to-speech conversion failed")
        logger.info("Heyyywed",tts_response.json().get("audios")[0])
        audio_base64 = tts_response.json().get("audios")[0]

        return {"audio": audio_base64}
    except Exception as e:
        logger.error(f"Error in text-to-speech endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)