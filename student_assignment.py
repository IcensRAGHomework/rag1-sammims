import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    system_prompt_str = """
        You are the assistant to provide holiday relevant query.
        For the query result, please porvide dates and names info in exactly structured JSON format as follows.
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}

        The language of the output result is same as the input.
    """

    messages = [
        SystemMessage(content = [{"type": "text", "text": system_prompt_str},]),
        HumanMessage(content = [{"type": "text", "text": question},]),
    ]

    model = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    response = model.invoke(messages)
    result = response.content
    return result
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message]).content
    
    return response

# print(demo("hello, 請使用繁體中文").content)
print(generate_hw01("2024年台灣10月紀念日有哪些?"))