from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model='gpt-5.2')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Please give detail report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Please summarize the given report into 5 lines {report}",
    input_variables=["report"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"Gen AI"})

print(result) 

chain.get_graph().print_ascii()