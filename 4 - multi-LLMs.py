import os
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


print("\n\n")


def getLLaMa():
    return ChatOllama(
        model="llama3"
    )

def getGPT4():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

memory = ConversationBufferMemory(memory_key="chatHistory", return_messages=True)

LLaMaPrompt = PromptTemplate(
    input_variables=["chatHistory", "input"],
    template=(
        "Você é um assistente útil alimentado por um modelo de IA leve. "
        "Mantenha suas respostas concisas e eficientes.\n\n"
        "Histórico da conversa:\n{chatHistory}\n\n"
        "Humano: {input}\nIA:"
    )
)

GPT4Prompt = PromptTemplate(
    input_variables=["chatHistory", "input"],
    template=(
        "Você é um assistente útil alimentado por um modelo de IA avançada. "
        "Mantenha suas respostas concisas e eficientes.\n\n"
        "Histórico da conversa:\n{chatHistory}\n\n"
        "Humano: {input}\nIA:"
    )
)

def createChains(LLM, prompt):
    return ConversationChain(llm=LLM, memory=memory, prompt=prompt, verbose=True)


def main():

    LLaMaChaining = createChains(getLLaMa(), LLaMaPrompt)
    GPT4Chaining = createChains(getGPT4(), GPT4Prompt)

    query = input("\nO que gostaria de saber: ")
    
    while query!="":

        if len(query.split()) <= 10:
            print("\nUsing LLaMA 3\n")
            resposta = LLaMaChaining.run({"input": query})
        else: 
            print("\nUsing GPT-4\n")
            resposta = GPT4Chaining.run({"input": query})
        
        print(f"\Resultado:\n{resposta}")
        query = input("\n\nO que mais gostaria de saber: ")

if __name__ == "__main__":
    main()
