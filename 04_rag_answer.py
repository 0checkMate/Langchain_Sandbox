from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter
import random
import re
import time

load_dotenv()

EMBEDDING_MODEL = 'gemini-embedding-001'
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'sop_documents'
LLM_MODEL = 'gemini-3.7-flash'
TOP_K = 3

#Questions to ask the RAG system. You can modify these to test different queries.
QUESTIONS = [
    'How long should surfaces be sanitised for?',
    'What temperature must juice be stored at, and for how long?',
    'What is form QA-12 used for?',
    'What is the company holiday policy?',
]

# System rules for the LLM to follow when generating answers. This ensures that the model only uses the provided context and does not hallucinate information.
SYSTEM_RULES = (
    'You answer questions about company procedures. '
    'Use ONLY the context provided. Do not use outside knowledge. '
    'If the context does not contain the answer, reply exactly: '
    'I could not find this in the provided documents. '
    'Otherwise answer concisely and name the section number(s) you relied on.'
)

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_RULES),
    ('human', 'Context:\n{context}\n\nQuestion: {question}'),
])

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# Add a rate limiter to avoid exceeding the API's request limits
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.07,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

# Initialize the LLM with the specified model and rate limiter
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, max_retries=3, rate_limiter=rate_limiter)

# Create a chain that combines the prompt, LLM, and output parser. This chain will take the context and question as input and produce a concise answer based on the provided documents.
chain = prompt | llm | StrOutputParser()

# Function to format the retrieved documents into a string for the prompt
def format_context(docs):
    parts = []
    for doc in docs:
        label = f'[{doc.metadata["source"]}, position {doc.metadata["start_index"]}]'
        parts.append(f'{label}\n{doc.page_content}')
    return '\n\n'.join(parts)

# Retry logic for transient errors
MAX_ATTEMPTS = 20
BASE_WAIT = 10
MAX_WAIT = 120

TRANSIENT_MARKERS = ('429', '503', 'RESOURCE_EXHAUSTED', 'UNAVAILABLE')

def is_transient(error):
    text = str(error)
    return any(marker in text for marker in TRANSIENT_MARKERS)


def wait_time(error, attempt):
    match = re.search(r'retry in ([\d.]+)s', str(error))
    if match:
        return float(match.group(1)) + 2
    delay = min(BASE_WAIT * 2 ** (attempt - 1), MAX_WAIT)
    return delay + random.uniform(0, 3)


def invoke_with_retry(chain, inputs):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return chain.invoke(inputs)
        except Exception as error:
            if not is_transient(error) or attempt == MAX_ATTEMPTS:
                raise
            pause = wait_time(error, attempt)
            print(f'  Attempt {attempt} failed ({type(error).__name__}). Waiting {pause:.0f}s...')
            time.sleep(pause)

# Run the RAG query for each question
for question in QUESTIONS:
    print(f'QUESTION: {question}')

    try:
        docs = vector_store.similarity_search(question, k=TOP_K)
        answer = invoke_with_retry(
            chain, {'context': format_context(docs), 'question': question}
        )
    except Exception as error:
        print(f'FAILED: {type(error).__name__}: {str(error)[:150]}')
        print('=' * 60)
        continue

    print(f'ANSWER: {answer}')
    print('SOURCES RETRIEVED:')
    for doc in docs:
        print(f' - {doc.metadata["source"]}, position {doc.metadata["start_index"]}')
    print('=' * 60)