import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
smithery_api_key=os.getenv("SMITHERY_API_KEY")
smithery_profile=os.getenv("SMITHERY_PROFILE")