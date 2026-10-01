from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown


load_dotenv()

PRIMARY_MODEL = 'gemini-3.7-flash'
FALLBACK_MODEL = 'gemini-3.6-flash'

primary = ChatGoogleGenerativeAI(model=PRIMARY_MODEL, max_retries=3, temperature=0)
fallback = ChatGoogleGenerativeAI(model=FALLBACK_MODEL, max_retries=3, temperature=0)

llm = primary.with_fallbacks([fallback])

prompt = ChatPromptTemplate.from_template('Explain {topic} to a 10-year-old.')
chain = prompt | llm | StrOutputParser()

response = chain.invoke({'topic': 'quantum physics'})
Console().print(Markdown(response))