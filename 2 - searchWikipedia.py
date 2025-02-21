import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.retrievers import WikipediaRetriever


print("\n\n")


def getOpenAiLLM():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# --- Funcionalidade da API do Wikipedia para ser chamada durante o funcionamento do agente ---
def getTools():
    wikipediaTool = Tool(
        name="Wikipedia",
        func=lambda query: WikipediaRetriever(lang="pt").invoke(query),
        description="Procure na Wikipedia por informações relavantes."
    )
    return [wikipediaTool]

def main():
    LLM = getOpenAiLLM()
    tools = getTools()

    agent = initialize_agent(
        tools=tools,
        llm=LLM,
        agent="zero-shot-react-description"
    )

    query = input("O que gostaria de buscar na Wikipedia: ")

    while query!="":
        resultado = agent.run(query)
        print("\nResultado:\n", resultado)
        query = input("\n\nPosso lhe ajudar em mais algo: ")


if __name__ == "__main__":
    main()