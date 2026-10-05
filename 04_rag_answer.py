from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

EMBEDDING_MODEL = 'gemini-embedding-001'
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'sop_documents'
LLM_MODEL = 'gemini-3.7-flash'
TOP_K = 3

QUESTIONS = [
    'How long should surfaces be sanitised for?',
    'What temperature must juice be stored at, and for how long?',
    'What is form QA-12 used for?',
    'What is the company holiday policy?',
]

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

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.07,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, max_retries=3, rate_limiter=rate_limiter)

chain = prompt | llm | StrOutputParser()

def format_context(docs):
    parts = []
    for doc in docs:
        label = f'[{doc.metadata["source"]}, position {doc.metadata["start_index"]}]'
        parts.append(f'{label}\n{doc.page_content}')
    return '\n\n'.join(parts)


for question in QUESTIONS:
    docs = vector_store.similarity_search(question, k=TOP_K)
    answer = chain.invoke({'context': format_context(docs), 'question': question})

    print(f'QUESTION: {question}')
    print(f'ANSWER: {answer}')
    print('SOURCES RETRIEVED:')
    for doc in docs:
        print(f' - {doc.metadata["source"]}, position {doc.metadata["start_index"]}')
    print('=' * 60)