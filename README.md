Building the Vector Database Step-by-Step
1. Installing and Setting Up the Environment
We start by ensuring all necessary libraries are installed:

python
Copy
Edit
!pip install langchain langchain-openai langchain-core pypdf
We also set up environment variables securely using:

python
Copy
Edit
from dotenv import load_dotenv
load_dotenv()
This loads our API keys and configuration details.

2. Loading the PDF Documents
To extract data from PDFs, we utilize PyPDFDirectoryLoader from langchain_community:

python
Copy
Edit
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("/content/")
documents = loader.load()
print(f"Loaded {len(documents)} documents")
Each page of the PDF becomes a document that we can later embed.

3. Splitting Documents for Better Embeddings
Since PDFs can be long, we split them into smaller chunks for better semantic embeddings:

python
Copy
Edit
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)
This step ensures that embeddings capture meaningful context.

4. Creating Embeddings Using OpenAI
Next, we generate vector embeddings for our text chunks:

python
Copy
Edit
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
These embeddings numerically represent the semantic meaning of each text chunk.

5. Storing Data in a Vector Store
To efficiently store and search the embeddings, we use a vector database like FAISS or Chroma:

python
Copy
Edit
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)
This sets up a searchable vector index over all our PDF data!

6. Performing Smart Semantic Search
Finally, we can now query the vector database:

python
Copy
Edit
query = "What are the key concepts of transformers?"
matching_docs = db.similarity_search(query)

for doc in matching_docs:
    print(doc.page_content[:500])
Instead of keyword search, the system retrieves based on meaning!

Summary
Through this project, we have transformed static PDFs into a dynamic, intelligent search system using the power of vector embeddings and AI models.
By leveraging LangChain and OpenAI, we created a pipeline that:

Extracts text from documents,

Splits them into digestible chunks,

Embeds them into a vector space,

And enables semantic querying across the knowledge base.

This forms the foundation for building applications like document QA bots, RAG-powered assistants, and smart search engines.

In the era of AI, connecting information to users meaningfully is the ultimate superpower — and with LangChain + OpenAI, it's now within everyone's reach.
