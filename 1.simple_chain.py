from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt_template = PromptTemplate(
    template= "Give me the summary on  {topic}",
    input_variables= ["topic"]
)

model = ChatOpenAI(model='gpt-5.2')

parser = StrOutputParser()

chain = prompt_template | model | parser

result = chain.invoke(input={"topic":"Gen AI"})

print(result)
