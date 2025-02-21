import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
#from langchain_ollama import ChatOllama


print("\n\n")


# --- Informação base ---
baseDeDados = """
Asimov é considerado um dos mestres da ficção científica e, junto com Robert A. Heinlein e Arthur C. Clarke, foi considerado um dos "três grandes" dessa área da literatura. A obra mais famosa de Asimov é a Série da Fundação, também conhecida como Trilogia da Fundação, que faz parte da série do Império Galáctico e que logo combinou com a Série Robôs. Também escreveu obras de mistério e fantasia, assim como uma grande quantidade de não-ficção. No total, escreveu ou editou mais de 500 volumes, aproximadamente 90 000 cartas ou postais, e tem obras em cada categoria importante do sistema de classificação bibliográfica de Dewey, exceto em filosofia.[1]

A maioria de seus livros mais populares sobre ciência, explicam conceitos científicos de uma forma histórica, voltando no tempo o mais longe possível, quando a ciência em questão estava nos primeiros estágios. Ele providência, muitas vezes, datas de nascimento e falecimento dos cientistas que menciona, também etimologias e guias de pronunciação para termos técnicos. Alguns exemplos incluem, "Guide to Science", os três volumes de "Understanding Physics" e a "Chronology of Science and Discovery", e trabalhos sobre Astronomia, Matemática, a Bíblia, escritos de William Shakespeare e Química.

Em 1981, um asteroide recebeu seu nome em sua homenagem, o 5020 Asimov.

Isaac Asimov nasceu Isaak Yudovich Ozimov (em russo, Исаак Юдович Озимов) em Petrovichi, uma pequena cidade situada na então Subdivisão Governamental de Gomel, situada no oeste da República Soviética Russa (atual Oblast de Smolensk, Rússia), perto do que hoje é a fronteira entre a Federação Russa e a Bielorrússia. De origem russo-judaica, era filho de Judah Asimov, um comerciante e moleiro, e Anna Rachel Berman-Asimov, uma dona-de-casa, oriunda de uma tradicional família de judeus. Em virtude das diferenças entre o calendário hebraico e o calendário juliano (à época ainda em uso na região pela Igreja Ortodoxa), bem como pela falta de registros, sua data de nascimento não pode ser precisada, situando-se entre 4 de outubro de 1919 e 2 de janeiro de 1920, sendo esta última considerada como a correta por Asimov, que sempre celebrou seu aniversário a 2 de janeiro. A família deriva seu nome de озимые (ozimiye), uma palavra da língua russa que significa um cereal de inverno que o seu bisavô negociava, ao qual o sufixo paterno foi adicionado. Sua família emigrou para os Estados Unidos quando ele tinha três anos de idade,[2][3] em 1923, se estabelecendo na cidade de Nova York. Como seus pais falavam sempre iídiche e inglês com ele, ele nunca aprendeu russo. Enquanto crescia no borough nova-iorquino de Brooklyn, Asimov aprendeu a ler, por si próprio, quando tinha cinco anos e permaneceu fluente em iídiche, bem como em inglês. Seus pais tinham uma loja de doces, e toda a gente da família tinha de lá trabalhar. Revistas baratas de papel de polpa, chamadas pulp sobre ficção científica eram vendidas em lojas, e ele começou a lê-las. Por volta dos onze anos, começou a escrever histórias próprias e, por volta dos dezenove anos, tendo-se tornado fã de ficção científica, começou a vender suas histórias a revistas. John W. Campbell, o editor de Astounding Science Fiction, para quem ele vendeu suas primeiras histórias, foi uma forte influência formativa e tornou-se um amigo. Nesta revista, publicou o conto Liar! (1941) onde apresentou Susan Calvin,[4] personagem que tornou-se recorrente em sua obra e, interpretada por diversas atrizes na TV e cinema, dentre as quais Bridget Moynahan, no filme de 2004, I, Robot.


A partir da esquerda: Robert A. Heinlein, L. Sprague de Camp e Asimov na Philadelphia Naval Shipyard em 1944.
Asimov foi aluno das New York City Public Schools (escolas públicas de Nova Iorque), inclusive a Boys' High School, de Brooklyn. A partir daí, ele foi para a Universidade Columbia, onde se graduou em 1939, depois tirando um Ph.D. em bioquímica, em 1948. Entretanto, passou três anos, durante a Segunda Guerra Mundial, a trabalhar como civil na Naval Air Experimental Station, do porto da Marinha em Filadélfia. Quando a guerra acabou, ele foi destacado para o Exército Americano, tendo só servido nove meses antes de ser honrosamente reformado. Durante sua breve carreira militar, ele ascendeu ao posto de cabo, baseado na sua habilidade para escrever à máquina, e escapou por pouco de participar nos testes da bomba atómica em 1946 no atol de Bikini.

Depois de completar seu doutorado, Asimov entrou na faculdade de Medicina da Universidade de Boston, com a qual permaneceu associado a partir daí. Depois de 1958, isto foi sem ensinar,[5] já que se virou para a escrita em tempo integral (suas receitas da escrita já excediam as do salário académico). Pertencer ao quadro permanente significou que ele manteve o título de professor associado e, em 1979, a universidade honrou sua escrita promovendo-o a professor catedrático de bioquímica. Os arquivos pessoais de Asimov, a partir de 1965, estão arquivados na Mugar Memorial Library da universidade, doados por ele a pedido do curador, Howard Gottlieb. A colecção preenche 464 caixas em 71 metros de prateleira.

Asimov foi membro e vice-presidente por muito tempo da Mensa, ainda que relutante: ele os descrevia como "intelectualmente combalidos". Exercia, com mais frequência e assiduidade, a presidência da American Humanist Association (Associação Humanista Americana).

Asimov casou-se com Gertrude Blugerman (Canadá, 1917 — Boston, 1990), em 26 de julho de 1942. Tiveram duas crianças, David (n. 1951) e Robyn Joan (n. 1955). Depois da separação, em 1970, ele e Gertrude divorciaram-se em 1973, e Asimov casou-se com Janet Jeppson mais tarde, no mesmo ano.[6]

Asimov era um claustrófilo; gostava de espaços pequenos fechados. No primeiro volume da sua autobiografia, ele conta um desejo infantil de possuir uma banca de jornais numa estação de metrô de Nova Iorque, dentro da qual ele se fecharia e escutaria o ruído dos carros enquanto lia.[7]

Asimov tinha medo de voar, só o tendo feito duas vezes na vida inteira (uma vez, durante seu trabalho na Naval Air Experimental Station, e outra, na volta para casa da base militar de Oahu, em 1946). Raramente viajava grandes distâncias, em parte por causa de sua aversão a voar, adicionada às dificuldades logísticas de viajar longas distâncias. Esta fobia influenciou várias das suas obras de ficção, como as histórias de mistério de Wendell Urth e as novelas sobre robôs de Elijah Baley. Nos seus últimos anos, ele gostava de viajar em navios de cruzeiro e, em várias ocasiões, ele fez parte do "entretenimento" no cruzeiro, dando palestras baseadas em ciência, em navios, como os RMS Queen Elizabeth 2. Asimov sabia entreter muitíssimo bem, era prolífico e procurado como discursador. Seu sentido de tempo era fantástico; ele nunca olhava para um relógio, mas invariavelmente falava precisamente o tempo combinado.

Asimov era um participante habitual em convenções de ficção científica, onde ficava amável e disponível para conversa. Ele respondia pacientemente a dezenas de milhares de perguntas e outro tipo de correio com postais, e gostava de dar autógrafos. Embora gostasse de mostrar seu talento, raramente parecia levar-se a si próprio demasiadamente a sério.

Ele era de altura mediana, forte, com bigode e um óbvio sotaque de judeus do Brooklyn. Sua motoridade física era bastante limitada. Ele nunca aprendeu a nadar ou andar de bicicleta; no entanto, aprendeu a conduzir um carro, depois de se mudar para Boston. No seu livro de humor, Asimov Laughs Again, ele descreve a condução em Boston como "anarquia sobre rodas".

Ele demonstrou seu amor por conduzir, em seu conto de ficção científica, Sally, sobre carros-robôs. Um leitor atento reparará que ele faz uma descricção detalhada de um dos carros a que chama 'Giuseppe', de Milão - o que significa que Giuseppe era um Alfa Romeo. Asimov não especificou nenhum outro tipo de veículo em nenhuma das suas histórias, o que levou muitos fãs a considerarem que ele foi contratado por aquela marca de automóvel.

Os interesses variados de Asimov incluíram, nos seus anos tardios, sua participação em organizações devotadas à opereta de Gilbert and Sullivan e em The Wolfe Pack, um grupo de seguidores dos mistérios de Nero Wolfe, escritos por Rex Stout. Ele era um membro proeminente da Baker Street Irregulars, a mais importante sociedade sobre Sherlock Holmes. De 1985 até sua morte em 1992, ele foi presidente da American Humanist Association; seu sucessor foi o amigo e congênere escritor, Kurt Vonnegut. Ele também era um amigo próximo do criador de Star Trek, Gene Roddenberry, e foram-lhe dados créditos em Star Trek: The Motion Picture, pelos conselhos que deu durante a produção.

Asimov morreu em 6 de abril de 1992 em Nova Iorque. Foi cremado e suas cinzas foram espalhadas. Ele deixou sua segunda mulher, Janet, e as crianças do primeiro casamento. Dez anos depois da sua morte,[8] a edição da autobiografia de Asimov, It's Been a Good Life, revelou que sua morte foi causada por SIDA (br: AIDS); ele contraiu o vírus HIV através de uma transfusão de sangue recebida durante a operação de bypass em Dezembro de 1983.[9] A causa específica da morte foi falha cardíaca e renal, como complicações da infecção com o vírus da SIDA.

Janet Asimov escreveu no epílogo de It's Been a Good Life que Asimov o teria querido tornar público, mas seus médicos convenceram-no a permanecer em silêncio, avisando que o preconceito antiSIDA estender-se-ia a seus familiares. A família de Asimov considerou divulgar sua doença antes de ele morrer, mas a controvérsia que ocorreu, quando Arthur Ashe divulgou que ele tinha SIDA, convenceu-os do contrário. Dez anos mais tarde, depois da morte dos médicos de Asimov, Janet e Robyn concordaram que a situação em relação à SIDA podia ser levada a público.[10]

É tio do crítico e jornalista Eric Asimov.[11]
"""

# --- Template com informação dinâmica ---
templateDynamic="""
    Dada uma informação {info} sobre uma pessoa, eu quero que você crie:
    1. um resumo curto
    2. dois fatores interessantes sobre ele
"""

# --- Criação do objeto do template ---
templateObject = PromptTemplate(input_variables=["info"], template=templateDynamic)

# --- Objeto do LLM ---
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
#LLM = ChatOllama(model_name="llama3")

# --- Encadeamento ---
chaining = templateObject | LLM | StrOutputParser()
resposta = chaining.invoke(input={"info": baseDeDados})

print(resposta)