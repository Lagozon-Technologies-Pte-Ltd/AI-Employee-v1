import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")
smithery_api_key=os.getenv("SMITHERY_API_KEY")
smithery_profile=os.getenv("SMITHERY_PROFILE")
azure_endpoint=os.getenv("AZURE_ENDPOINT")
azure_api_version=os.getenv("AZURE_API_VERSION")
azure_subscription_key=os.getenv("AZURE_SUBSCRIPTION_KEY")
azure_deployment=os.getenv("AZURE_DEPLOYMENT")  