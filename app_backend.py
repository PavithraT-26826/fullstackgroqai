import os
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GroqBackend")

class GeminiChatEngine:  # Kept as GeminiChatEngine so app_frontend.py doesn't break
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment or .env file.")
        
        # Connect to Groq Client
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name or "llama-3.3-70b-versatile"
        logger.info(f"Initialized Groq Engine with model: {self.model_name}")

    def generate_chat_response(
        self, 
        message_history: List[Dict[str, str]], 
        user_message: str, 
        system_instruction: str = "You are a helpful, precise AI assistant."
    ) -> Tuple[str, Dict[str, int]]:
        try:
            # Format messages for Groq/OpenAI structure
            messages = [{"role": "system", "content": system_instruction}]
            for msg in message_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            # Call Groq Llama 3.3 Model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )

            reply_text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens if response.usage else 0

            return reply_text, {"total_tokens": total_tokens}

        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"Backend Error: {str(e)}", {"total_tokens": 0}