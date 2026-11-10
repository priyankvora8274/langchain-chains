from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()
model = ChatOpenAI(model='gpt-5.2')

notes_prompt = PromptTemplate(
    template="Give me the simple and consice notes on given {report}",
    input_variables=["report"]
)

quiz_prompt = PromptTemplate(
    template="Create Quiz contains 5 simple questions from the given {report}",
    input_variables=["report"]
)

merge_prompt = PromptTemplate(
    template="Merge give notes and Quiz into one document Notes: {notes} & Quiz: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain  = RunnableParallel(
    {
        'notes': notes_prompt | model | parser,
        'quiz': quiz_prompt | model | parser
    }
)

merge_chain = merge_prompt | model | parser

chain = parallel_chain  | merge_chain

text = """
Introduction
Generative Artificial Intelligence (Gen AI) is one of the most advanced and rapidly growing technologies in the modern world. It is a branch of artificial intelligence that focuses on creating new content such as text, images, videos, audio, and computer code. Unlike traditional AI systems that mainly analyze data and make predictions, Generative AI can produce new information based on patterns learned from large datasets. Popular examples of Generative AI include ChatGPT, Google Gemini, Claude, and DALL·E. These technologies are transforming how people communicate, learn, work, and solve problems.
How Generative AI Works
Generative AI works through advanced machine learning algorithms and deep learning models. These models are trained using large amounts of data collected from various sources. During training, AI systems learn patterns, relationships, and structures within the data. Large Language Models (LLMs), such as those used in ChatGPT, can understand and generate human-like text. Transformer architectures help these models process information and recognize relationships between words. When a user provides a question or instruction, known as a prompt, the AI generates a response based on its training and the available context.
Applications of Generative AI
Generative AI is widely used in different industries, including education, healthcare, finance, entertainment, and software development. In education, AI tools help students understand complex topics, summarize information, and improve their learning experience. In healthcare, Generative AI assists professionals with medical documentation, research, and administrative tasks. Businesses use AI-powered chatbots to improve customer service and automate repetitive activities. In software engineering, developers use tools such as GitHub Copilot to generate code, identify errors, and improve productivity. Marketing companies also use Generative AI to create advertisements, social media content, and promotional materials.
"""

result = chain.invoke({"report":text})

print(result)

print(chain.get_graph().print_ascii())