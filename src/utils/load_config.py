import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
from pymongo import MongoClient
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import chromadb

# Load environment variables
print("Environment variables loaded:", load_dotenv())

class LoadConfig:
    def __init__(self) -> None:
        """Initialize configuration and services."""
        # Load config file
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Load MongoDB Cloud configuration
        self.load_mongodb_config()

        # Load LLM and RAG configurations
        self.load_llm_configs(app_config)
        self.load_openai_models()
        self.load_chroma_client()
        self.load_rag_config(app_config)

    def load_mongodb_config(self):
        """Loads MongoDB Cloud connection details from environment variables."""
        try:
            mongodb_uri = os.getenv("MONGODB_URI")  # Remote MongoDB Cloud URI
            if not mongodb_uri:
                raise ValueError("❌ Missing MongoDB Cloud URI in environment variables.")

            # Initialize MongoDB Cloud client
            self.mongodb_client = MongoClient(mongodb_uri)
            self.mongodb_database = "jobs"  # Change if necessary
            self.mongodb_collection = "job_posting"
            
            db = self.mongodb_client[self.mongodb_database]
            collection = db[self.mongodb_collection]

            
            # Ensure text index is created
            collection.create_index(
                [("Job Id", "text"),
                ("Experience", "text"),
                ("Qualifications", "text"),
                ("Salary Range", "text"),
                ("location", "text"),
                ("Country", "text"),
                ("Work Type", "text"),
                ("Preference", "text"),
                ("Contact Person", "text"),
                ("Job Title", "text"),
                ("Role", "text"),
                ("Job Portal", "text"),
                ("Job Description", "text"),
                ("Benefits", "text"),
                ("skills", "text"),
                ("company", "text"),
                ("Responsibilities", "text"),
                ("Company", "text")
                ], name="job_text_index"
            )    
                

            print("✅ Successfully connected to MongoDB Cloud and ensured text index!")


        

        except Exception as e:
            raise ValueError(f"❌ Error connecting to MongoDB Cloud: {e}")

    def load_llm_configs(self, app_config):
        """Loads LLM configurations."""
        self.model_name = os.getenv("GPT_DEPLOYMENT_NAME")
        self.agent_llm_system_role = app_config["llm_config"]["agent_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.embedding_model_name = os.getenv("EMBED_DEPLOYMENT_NAME")

    def load_openai_models(self):
        """Loads Azure OpenAI configurations."""
        try:
            azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_openai_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
            azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            gpt_deployment_name = os.getenv("GPT_DEPLOYMENT_NAME")

            if not all([azure_openai_api_key, azure_openai_endpoint, azure_openai_api_version, gpt_deployment_name]):
                raise ValueError("❌ Missing required Azure OpenAI environment variables.")

            # Initialize Azure OpenAI client
            self.azure_openai_client = AzureOpenAI(
                api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version,
            )

            # Initialize LangChain LLM
            self.langchain_llm = AzureChatOpenAI(
                azure_endpoint=azure_openai_endpoint,
                azure_deployment=gpt_deployment_name,
                model=gpt_deployment_name,
                openai_api_version=azure_openai_api_version,
                temperature=self.temperature
            )

        except KeyError as e:
            raise ValueError(f"❌ Missing required environment variable: {e}")

    def load_chroma_client(self):
        """Initializes ChromaDB client."""
        self.chroma_client = chromadb.PersistentClient(path=str(here("chromadb_data")))

    def load_rag_config(self, app_config):
        """Loads RAG configurations."""
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]
