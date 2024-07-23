import os
import openai
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

# Initialize OpenAI with your API key
openai.api_key = OPENAI_API_KEY

# Function to read PDF documents
def readdoc(directory):
    try:
        print(f"Reading documents from directory: {directory}")
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return []
        
        filereader = PyPDFDirectoryLoader(directory)
        documents = filereader.load()
        print(f"Number of documents loaded: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# Function to chunk documents
def chunkdata(doc, chunk_size=800, chunk_overlap=50):
    try:
        if not doc:
            print("No documents to chunk.")
            return []
        
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = textsplitter.split_documents(doc)
        print(f"Number of chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

# Correct path to the directory containing your PDF files
pdf_directory = r"C:\Users\HP\Desktop\Business studies"

# Load and chunk documents
documents = readdoc(pdf_directory)
chunked_documents = chunkdata(documents)

# Initialize embeddings with your API key
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Print embeddings object for debugging
print(f'EMBEDDINGS: {embeddings}')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create Pinecone index
index_name = "langchainvector"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Assuming 1536 is the dimensionality of the embeddings

index = Pinecone.from_documents(chunked_documents, embeddings, index_name=index_name)

# Function to retrieve matching documents for a query
def retrievequery(query, k=2):
    matchingresults = index.similarity_search(query, k=k)
    return matchingresults

# Import necessary modules for QA chain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

# Initialize the OpenAI LLM and QA chain
llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
chain = load_qa_chain(llm, chain_type="stuff")

# Function to retrieve answer for a query
def retrieveanswer(query):
    doc_search = retrievequery(query)
    print(f"Document Search Results: {doc_search}")
    response = chain.run(input_documents=doc_search, question=query)
    return response

# Example usage
query = "What are the key strategies for business growth?"
answer = retrieveanswer(query)
print(f"Answer: {answer}")
