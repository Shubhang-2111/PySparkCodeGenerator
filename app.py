import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Sample JSON data
sample_json = {
    "employees": [
        {"id": 1, "name": "Alice", "department": "Sales", "salary": 50000},
        {"id": 2, "name": "Bob", "department": "Engineering", "salary": 70000}
    ]
}

# Business logic to apply
business_logic = "Flag Employees Who Earn Below Department Average Salary"

# Prompt template
template = """
You are a senior PySpark developer.

Given the following JSON data:
{json_data}

And the following business logic:
"{logic}"

Generate only the PySpark code that implements the logic. Do not include any explanation, comments, or extra text. Return only the code.
"""


# Format the prompt
prompt = ChatPromptTemplate.from_template(template)

# Initialize Groq + DeepSeek LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
)

# Set up output parser
parser = StrOutputParser()

# Chain the steps: prompt → LLM → output parser
chain = prompt | llm | parser

# Invoke the chain
response = chain.invoke({
    "json_data": json.dumps(sample_json, indent=2),
    "logic": business_logic
})

print("Generated PySpark Code:\n")
print(response)
