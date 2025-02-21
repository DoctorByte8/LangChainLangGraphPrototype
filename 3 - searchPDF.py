import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


print("\n\n")


# ---- Carregar o PDF ----
def loadPDF(filePath):
    loader = PyPDFLoader(filePath)
    documents = loader.load()
    return documents

# ---- Quebrar o PDF em pedaços para vetoriza-lo ----
def splitDocuments(documents):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return textSplitter.split_documents(documents)

# Vetorização dos pedações, usando o tranformer da OpenAI
def createVectorStorage(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorStorage = FAISS.from_documents(chunks, embeddings)
    return vectorStorage

# Chaining usando a base vetorizada como retriever
def createChains(vectorStorage):
    retriever = vectorStorage.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    LLM = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chaining = RetrievalQA.from_chain_type(llm=LLM, retriever=retriever)
    return chaining

def main():
    filePath = "PRFol_215985_Os investimentos da Petrobras.pdf"
    
    documents = loadPDF(filePath)
    chunks = splitDocuments(documents)
    vectorStorage = createVectorStorage(chunks)

    
    chaining = createChains(vectorStorage)
    
    print("\nPDF carregado. Agora faça perguntas sobre o conteúdo do PDF.")
    query = input("\nO que gostaria de saber: ")
    
    while query!="":
        answer = chaining.run(query)
        print(f"\nResultado:\n{answer}")
        query = input("\nO que mais gostaria de saber: ")

if __name__ == "__main__":
    main()
