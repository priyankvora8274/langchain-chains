from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_classic.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model='gpt-5.2')

class Feedback(BaseModel):
    sentiment:  Literal["Positive", "Negative"] = Field(description="Sentiment of the customer feedback")
    
parser = PydanticOutputParser(pydantic_object=Feedback)
strParser = StrOutputParser()

prompt = PromptTemplate(
    template="Classify the given feedback from the customer into Positive or Negative Sentiment {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain1 = prompt | model | parser

prompt1 = PromptTemplate(
    template="Please respond appropriately to this postive feedback of customer as professional {feedback}",
    input_variables=["feedback"]
)
prompt2 = PromptTemplate(
    template="Please respond  appropriatelyto this negative feedback of customer as professional {feedback}",
    input_variables=["feedback"]
)

condt_chain = RunnableBranch(
    (lambda x:x.sentiment == "Positive", prompt1 | model | strParser ),
    (lambda x:x.sentiment == "Negative", prompt2 | model | strParser ),
    RunnableLambda(lambda x: "Did not classifly in sentiment")
)

chain = chain1 | condt_chain

result = chain.invoke({"feedback":"This smartphone has very good camera"})

print(result)

print(chain.get_graph().print_ascii())