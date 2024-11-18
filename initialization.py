from langchain.llms import openai
from langchain import LLMChain, PromtTemplate
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory

from connect_db import session

OPENAI_API_KEY = ""
ASTRADB_KEYSPACE = "cyoa_db"

message_history= CassandraChatMessageHistory(
    session_id = 'anything',
    session = session,
    keyspace = ASTRADB_KEYSPACE,
    ttl_seconds = 3600
)

message_history.clear()

cass_message_buffer = ConversationBufferMemory(
    memory_key="chat_memory",
    chat_memory=message_history
)

llm = openai.OpenAI(openai_api_key = OPENAI_API_KEY)