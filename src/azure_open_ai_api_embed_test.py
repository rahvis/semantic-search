import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Ensure environment variables are loaded correctly
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embedding_model = os.getenv("EMBED_DEPLOYMENT_NAME")

# Debugging: Print environment variables to check if they are loaded
if not all([api_key, azure_endpoint, api_version, embedding_model]):
    raise ValueError("❌ Missing environment variables! Check .env file.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Sample text to embed
sample_text = "This is a test sentence for embedding."

try:
    # Generate embedding
    response = client.embeddings.create(
        input=sample_text,
        model=embedding_model  # Ensure this matches your Azure deployment name
    )

    # Print the embedding vector
    print("✅ Embedding Vector:", response.data[0].embedding)
    print("✅ Embedding Length:", len(response.data[0].embedding))

except Exception as e:
    print("❌ Error generating embedding:", e)
