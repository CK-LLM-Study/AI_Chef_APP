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

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=API_KEY, temperature=0.0)
# llm = OpenAI(openai_api_key=API_KEY, temperature=0.9)

meal_template = """Give me the ingredients and each quantities using the meal below.
Be sure to write in Korean. 
Meal : {meal}

[Example]
- 카레 100g
- 쌀 300g
- 사과 2개       
- 돼지고기 100g
- 참기름 두수푼
...
"""

# meal template
meal_template = PromptTemplate(
    input_variables=["meal"],
    template=meal_template
)
                          
cooking_template = """Give me the detail cooking steps using the ingredients below. 
Summarize 10 cooking steps. Be sure to write in Korean. Write cooking stpe lin by line. 
Ingredients : {ingredients}

Example :
Output:
- step 1. 밥 500g을 물과 함께 끓이기 시작한다.
- step 2. 돼지고기 100g을 넣고 물과 함께 데치다.
- step 3. 된장 약간을 넣고 끓여준다.
- step 4. 달걀 한개를 넣고 끓이고 잔향을 낸다.
- step 5. 김 4장을 넣고 조리용 소금을 뿌리고 반숙시킨다.
- step 6. 파 2장과 무 2개를 넣고 끓인다.
...
"""

cooking_template = PromptTemplate(
    input_variables=["ingredients"],
    template=cooking_template
)

# print(meal_chain.run("비빔밥"))

# Memory
ingredients_memory = ConversationBufferMemory(input_key='meal', memory_key='chat_history')
step_memory = ConversationBufferMemory(input_key='ingredients', memory_key='chat_hostory')

meal_chain = LLMChain(
    llm=llm,
    prompt=meal_template,
    output_key="ingredients",
    memory=ingredients_memory,  
    verbose=True
)

cooking_chain = LLMChain(
    llm=llm,
    prompt=cooking_template,
    output_key="steps",
    memory=step_memory,  
    verbose=True
)

# print(cooking_chain.run(["사과", "빵"]))

overall_chain = SequentialChain(
    chains=[meal_chain, cooking_chain],
    input_variables=["meal"],
    output_variables=["ingredients", "steps"],
    verbose=True
)

# print(overall_chain({"meal":"비빔밥"}))

st.title("🧑‍🍳 ChatGPT AI Chef")
user_prompt = st.text_input("Ask Your Recipies")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({'meal': user_prompt})
        print(output)

        col1, col2 = st.columns(2)
        col1.write(output['ingredients'])
        col2.write(output['steps'])
        
        with st.expander('Meals History'):
            st.info(ingredients_memory.buffer)
            
        with st.expander('Cooking History'):
            st.info(step_memory.buffer)