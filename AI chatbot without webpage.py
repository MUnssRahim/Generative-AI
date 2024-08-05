import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import openai
from langchain.embeddings import OpenAIEmbeddings
import pinecone

# Load environment variables
load_dotenv()
GoogleAPIKEY = os.getenv("Google_API_KEY")
OpenAIAPIKEY = os.getenv("OpenAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Configure API keys
genai.configure(api_key=GoogleAPIKEY)
openai.api_key = OpenAIAPIKEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

index_name = "pdf-chat-index"

# Ensure the index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=512)  # Adjust dimension as needed

index = pinecone.Index(index_name)

def getpdfdirect(file_path):
    """Extract text from a PDF file given its file path."""
    text = ""
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def texttochunk(text, chunk_size=10000, chunk_overlap=50):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def getconversation(prompt_template, model_name="gemini-pro", temperature=0.5, provider="google"):
    """Create a conversation chain based on the provider."""
    if provider == "google":
        model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        model = openai.Completion.create(model=model_name, temperature=temperature)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'questions'])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def storePinecone(text_chunks, model_name="text-embedding-v1", provider="google"):
    """Store text chunks in Pinecone."""
    if provider == "google":
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        embeddings = OpenAIEmbeddings(model=model_name)
    
    for i, chunk in enumerate(text_chunks):
        embedding = embeddings.embed_documents([chunk])[0]
        index.upsert([(str(i), embedding, {"text": chunk})])

def userinput(userquestion, chain, provider="google"):
    """Process user question and get the response from the chain."""
    if provider == "google":
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-v1")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-v1")
    
    user_embedding = embeddings.embed_query(userquestion)
    results = index.query(user_embedding, top_k=5, include_metadata=True)
    
    docs = [{"text": match["metadata"]["text"], "score": match["score"]} for match in results["matches"]]
    response = chain({"input_documents": docs, "question": userquestion}, return_only_outputs=True)
    return response["output_text"]

def process_pdf(file_path, user_question, provider="google"):
    """Main function to process a PDF file and answer a question."""
    rawtext = getpdfdirect(file_path)
    text_chunks = texttochunk(rawtext, chunk_size=10000, chunk_overlap=50)
    storePinecone(text_chunks, provider=provider)
    
    prompt_template = "Given the following context: {context}\nAnswer the following questions: {questions}"
    chain = getconversation(prompt_template, provider=provider)
    
    answer = userinput(user_question, chain, provider=provider)
    return answer

# Example usage
if __name__ == "__main__":
    file_path = os.path.expanduser("~\\Desktop\\ML AI.pdf") 
    user_question = "What is the main topic of the document?"
    provider = "google"  # or "openai"
    answer = process_pdf(file_path, user_question, provider)
    print("Answer:", answer)
