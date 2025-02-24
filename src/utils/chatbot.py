import os
from typing import List, Tuple
from utils.load_config import LoadConfig
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import langchain

langchain.debug = True

APPCFG = LoadConfig()

class ChatBot:
    """
    A ChatBot class that interacts with MongoDB Cloud to perform Q&A.
    """

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Responds to a message using MongoDB Cloud as the data source.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message.
            chat_type (str): Specifies the type of interaction.
            app_functionality (str): Defines the chatbot's role.

        Returns:
            Tuple[str, List]: Updated chatbot conversation.
        """

        if app_functionality == "Chat":
            if chat_type == "Q&A with MongoDB":
                try:
                    # Connect to MongoDB Cloud Collection
                    collection = APPCFG.mongodb_client[APPCFG.mongodb_database][APPCFG.mongodb_collection]

                    # Perform full-text search on job title, role, and description
                    query = {
                        "$text": {
                            "$search": message
                        }
                    }
                    results = list(collection.find(query).limit(5))  # Fetch up to 5 matching jobs

                    if not results:
                        response = "No relevant job postings found."
                    else:
                        response = "\n\n".join(
                            [
                                f"📌 **{res.get('Job Title', 'N/A')}**\n"
                                f"🏢 Company: {res.get('Company', 'N/A')}\n"
                                f"📍 Location: {res.get('location', 'N/A')}, {res.get('Country', 'N/A')}\n"
                                f"💰 Salary: {res.get('Salary Range', 'N/A')}\n"
                                f"📅 Posted on: {res.get('Job Posting Date', 'N/A')}\n"
                                f"📞 Contact: {res.get('Contact Person', 'N/A')} - {res.get('Contact', 'N/A')}\n"
                                f"🎓 Experience: {res.get('Experience', 'N/A')}\n"
                                f"📜 Qualifications: {res.get('Qualifications', 'N/A')}\n"
                                f"🛠 Work Type: {res.get('Work Type', 'N/A')}\n"
                                f"🎯 Preference: {res.get('Preference', 'N/A')}\n"
                                f"🌐 Job Portal: {res.get('Job Portal', 'N/A')}\n"
                                f"📌 Role: {res.get('Role', 'N/A')}\n"
                                f"💡 Skills: {res.get('skills', 'N/A')}\n"
                                f"📋 Responsibilities: {res.get('Responsibilities', 'N/A')}\n"
                                f"🎁 Benefits: {res.get('Benefits', 'N/A')}\n"
                                f"📝 Description: {res.get('Job Description', 'N/A')[:500]}..."  # Increased description length
                                for res in results
                            ]
                        )

                    # Format response using LLM
                    answer_prompt = PromptTemplate.from_template(APPCFG.agent_llm_system_role)
                
                    # Ensure all expected variables are included
                    chain = (
                        RunnablePassthrough.assign(result=lambda _: response)  # Ensure 'result' is provided
                        | answer_prompt
                        | APPCFG.langchain_llm
                        | StrOutputParser()
                    )
                    # Pass all required inputs to the chain
                    response = chain.invoke({"query": response, "question": message, "result": response})
                    
                    chatbot.append((message, response))  
                    return "", chatbot

                except Exception as e:
                    chatbot.append((message, f"❌ Error querying MongoDB Cloud: {e}"))
                    return "", chatbot

            else:
                chatbot.append((message, "Unsupported chat type."))
                return "", chatbot
        else:
            chatbot.append((message, "Invalid app functionality."))
            return "", chatbot
