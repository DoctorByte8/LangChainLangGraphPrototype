import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

print("\n\n\n\n")



information = """
Bruno, mais conhecido como Brunozor, Bruninzor ou apenas Brino, é um streamer, criador de conteúdo e sócio de uma organização de jogos.

Brino é um dos novos jogadores da série Ordem Paranormal, atuando como Mike no especial de Natal, Natal Macabro.

Mike (OPNM): Mike é um garoto pobrinho, que faz o melhor que pode como entregador de pizza para ajudar sua família do interior. Neste fim de ano, viajou para a cidade grande em busca de novas oportunidades.

Bruno foi chamado para Ordem Paranormal após um piada que ocorreu durante suas lives e nas redes sociais, em que ele insistia para participar de alguma forma no RPG. Cellbit viu as pessoas pedindo pela sua participação e, em referência a piada, descreveu Brino como "um pouquinho insistente" durante seu anúncio oficial como jogador em Natal Macabro.

Brino é um dos vários nomes encontrados nas lápides de Enigma do Medo, mais especificamente no Labirinto do Cemitério das Melodias. Em sua lápide tem a frase "Mais aleatório que a aleatoriedade." escrita.
"""

summaryTemplate="""
    Dada uma informação {information} sobre uma pessoa, eu quero que você crie:
    1. um resumo curto
    2. dois fatores interessantes sobre ele
"""

summaryPromptTemplate = PromptTemplate(input_variables=["information"], template=summaryTemplate)

LLM = ChatOpenAI(temperature=1.5, model_name="gpt-3.5-turbo")

chain = summaryPromptTemplate | LLM

res = chain.invoke(input={"information": information})

print(res)