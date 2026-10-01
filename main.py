from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = 'gemini-3.8-flash'

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
prompt = ChatPromptTemplate.from_template('Explain {topic} to a 10-year-old.')
chain = prompt | llm

response = chain.invoke({'topic': 'cold-press juice extraction'})
print(response.content)
