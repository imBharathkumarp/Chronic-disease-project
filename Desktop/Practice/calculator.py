from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

a = input("Enter a Number: ")
b = input("Enter a Number: ")

prompt = f"""
Do all the arithmetic operations on the numbers {a} and {b}.
"""
response = OllamaLLM(prompt)

print(response)
