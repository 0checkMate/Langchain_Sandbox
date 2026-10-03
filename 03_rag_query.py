from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = 'gemini-embedding-001'
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'sop_documents'
TOP_K = 2

QUESTIONS = [
    "What is the purpose of this document?",
    'How long should surfaces be sanitised for?',
    'How long should the juice last in the fridge?',
    'What happens if the juice gets warm?',
    'What is the company holiday policy?',
]

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

for question in QUESTIONS:
    print(f'\n\nQuestion: {question}')
    results = vector_store.similarity_search_with_score(question, k=TOP_K)

    for rank, (doc, score) in enumerate(results, start=1):
        start = doc.metadata.get('start_index', 'N/A')
        print(f' - Rank {rank}: {doc.metadata["source"]} | Score: {score:.3f} | starts at {start}')
        print(f'Content: {doc.page_content}\n')
        print('')
    print('=' * 60)