import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Retrieve credentials from .env
azure_openai_api_key = os.getenv("OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("OPENAI_API_BASE")
azure_openai_api_version = os.getenv("OPENAI_API_VERSION")
gpt_deployment_name = os.getenv("GPT_DEPLOYMENT_NAME")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint
)

# Test completion request
try:
    response = client.chat.completions.create(
        model=gpt_deployment_name,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "Hello! How are you?"}],
        temperature=0.7
    )
    
    print("Azure OpenAI API Test Successful!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("Error connecting to Azure OpenAI:", e)
