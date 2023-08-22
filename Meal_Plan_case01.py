from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.
API_KEY = os.environ['OPENAI_API_KEY']

from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# llm = OpenAI(openai_api_key=API_KEY, temperature=0.0)
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=API_KEY, temperature=0.0)

ml_template = """Give me 2 meal names and simple cooking steps that could be made using the following ingredients.
{ingredients}

Be sure to write in Korean. Write in line by line.
"""

# meal template
meal_template = PromptTemplate(
    input_variables=["ingredients"],
    template=ml_template,
)

# print(meal_template.format(ingredients="감자, 카레, 소고기"))

cal_template = """Give me the calories of the ingredients given below. 
{ingredients}
Be sure to write in Korean. Write in line by line.

Example :
Output:
- 돼지고기 (150g) 330 칼로리
- 감자 (1개) 130 칼로리
- 밥 (2공기) 360 칼로리
- 치즈 (적당량) 100 칼로리
...
"""

calories_template = PromptTemplate(
    input_variables=['ingredients'],
    template=cal_template
)

# Memory
meal_memory = ConversationBufferMemory(input_key='ingredients', memory_key='chat_history')
calories_memory = ConversationBufferMemory(input_key='ingredients', memory_key='chat_history')

meal_chain = LLMChain(
    llm=llm,
    prompt=meal_template,
    output_key="meals",  # the output from this chain will be called 'meals',
    memory=meal_memory,
    verbose=True
)

calorie_chain = LLMChain(
    llm=llm,
    prompt=calories_template,
    output_key="calories",  # the output from this chain will be called 'calories'
    memory=calories_memory,
    verbose=True
)

overall_chain = SequentialChain(
    chains=[meal_chain, calorie_chain],
    input_variables=["ingredients"],
    output_variables=["meals", "calories"],
    verbose=True
)

# overall_chain({"ingredients":"banana"})

st.title("🧑‍🍳 ChatGPT AI Chef")
user_prompt = st.text_input("A comma-separated list of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({'ingredients': user_prompt})

        col1, col2 = st.columns(2)
        col1.write(output['meals'])
        col2.write(output['calories'])
        
        with st.expander('Meals History'):
            st.info(meal_memory.buffer)
            
        with st.expander('Calorie History'):
            st.info(calories_memory.buffer)