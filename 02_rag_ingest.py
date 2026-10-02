from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = 'docs'
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Load all documents from the specified directory
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

# Split the documents into chunks
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