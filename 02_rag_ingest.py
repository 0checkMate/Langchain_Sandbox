import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Data ingest constants
DOCS_DIR = 'docs'
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Data embedding constants
EMBEDDING_MODEL = 'gemini-embedding-001'
CHROMA_DIR = 'chroma_db'
COLLECTION_NAME = 'sop_documents'

# Step 1 Load all documents from the specified directory
loader = DirectoryLoader(
    DOCS_DIR, 
    glob='**/*.txt', 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()

#print length of documents loaded (plus metadata, page content, characters)
print(f'Loaded {len(documents)} documents from {DOCS_DIR}')
for doc in documents:
    print(f' - {doc.metadata["source"]} ({len(doc.page_content)} characters)')  

# Step 2 Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)

#print chunk count and chunks
print(f'\nCreated {len(chunks)} chunks from {len(documents)} documents\n')
for i, chunk in enumerate(chunks):
    start = chunk.metadata.get('start_index', 'N/A')
    print(f' - Chunk {i+1}: {len(chunk.page_content)} characters | starts at {start}')
    print(chunk.page_content)
    print()

# Step 3 embed the chucnks and store them in Chroma
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Inspect one embedding before storing anything
sample_vector = embeddings.embed_query(chunks[0].page_content)
print(f'Embedding length: {len(sample_vector)} numbers')
print(f'First 5 values: {sample_vector[:5]}')

# Rebuild the index from scratch on every run
shutil.rmtree(CHROMA_DIR, ignore_errors=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)

print(f'\nStored {len(chunks)} chunks in {CHROMA_DIR}/')