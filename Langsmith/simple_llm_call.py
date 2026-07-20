from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate.from_template("{question}")

parser = StrOutputParser()

# Chain: prompt -> model -> parser
chain = prompt | model | parser

result = chain.invoke({"question": "What is the captial of India?"})
print(result)