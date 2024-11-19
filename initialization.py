from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import CassandraChatMessageHistory
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

from connect_db import connect_to_vector_db

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRADB_KEYSPACE = "cyoa_db"

template = """
You are now the guide of a mystical journey in the Whispering Woods. 
A traveler named Elara seeks the lost Gem of Serenity. 
You must navigate her through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Elara's fate. 

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

# session = connect_to_vector_db()

# message_history= CassandraChatMessageHistory(
#     session_id = 'anything',
#     session = session,
#     keyspace = ASTRADB_KEYSPACE,
#     ttl_seconds = 3600
# )

# message_history.clear()

conversation_history = {}

def update_conversation_history(role, message):
    global conversation_history
    conversation_history[len(conversation_history)] = {"role": role, "content": message}

def get_chat_history_string():
    global conversation_history
    history_str = ""
    for i in range(len(conversation_history)):
        history_str += f"{conversation_history[i]['role']}: {conversation_history[i]['content']}\n"
    return history_str

# Initialize the OpenAI LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Define the prompt template with proper parameters
prompt = PromptTemplate(
    template=template,
    input_variables=["chat_history", "human_input"]
)

# Start the game
while True:
    human_input = input("You: ")
    update_conversation_history("Human", human_input)

    prompt_with_history = prompt.format(
        chat_history=get_chat_history_string(),
        human_input=human_input
    )

    response = llm(prompt_with_history)
    update_conversation_history("AI", response)

    print(f"AI: {response}")

    # Check for end condition (e.g., "The End" in the response)
    if "The End" in response:
        break