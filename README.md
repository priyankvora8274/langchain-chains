# LangChain Chains 🔗

A collection of Python examples demonstrating how to build **LLM workflows with LangChain Expression Language (LCEL)** using simple, sequential, parallel, and conditional chains.

## 📂 Examples

| Script | What it does | Best use | Limitation |
|---|---|---|---|
| `1.simple_chain.py` | Generates a summary of a topic using `PromptTemplate → ChatOpenAI → StrOutputParser`. | Single-step text generation | Handles only one task at a time. |
| `2.sequential_chain.py` | Creates a detailed report, then summarizes it into five lines. | Multi-step workflows where one output feeds the next | Steps run in order, increasing latency. |
| `3.parallel_chain.py` | Generates concise notes and a five-question quiz from the same report, then merges both. | Independent tasks that can run concurrently | The final merge waits for both branches. |
| `4.conditional_chain.py` | Classifies customer feedback as positive or negative with Pydantic, then selects a response branch. | Sentiment-based routing and customer replies | Classification errors can send input down the wrong branch. |

## 🧠 Key Concepts

- **LCEL (`|`)** connects prompts, models, and parsers into reusable workflows.
- **`RunnableParallel`** runs independent branches on the same input.
- **`RunnableBranch`** selects a path based on a condition.
- **`PydanticOutputParser`** parses classification output into a typed model.
- **`get_graph().print_ascii()`** displays a chain's structure.

## 🛠️ Tech Stack

Python · LangChain · LangChain Core · LangChain Classic · OpenAI · Pydantic · python-dotenv

## 🚀 Getting Started

```bash
git clone https://github.com/priyankvora8274/langchain-chains.git
cd langchain-chains
pip install -r requirements.txt
```

Create a local `.env` file with your API key (do not commit it):

```env
OPENAI_API_KEY=your_api_key_here
```

Run an example:

```bash
python 1.simple_chain.py
```

The examples use `ChatOpenAI(model="gpt-5.2")`, so access to that model and an OpenAI API key are required.

## 🎯 Learning Goal

Understand when to **chain steps in sequence, run tasks in parallel, or route requests conditionally** to build modular AI applications.

> **Implementation note:** The conditional example currently passes the parsed `Feedback` object into prompts expecting `{feedback}`. You may need to preserve the original customer feedback alongside the sentiment so the response prompts receive the actual review text.
