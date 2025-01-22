import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate

from langchain import hub
from langchain_core.tools import tool
import requests
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import base64


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

CALENDARIFIC_API_URL = "https://calendarific.com/api/v2/holidays"
CALENDARIFIC_API_KEY = "ITsSHlCEb53LF4LEloRGX2Wfzv6mju73"

# @tool decorator: https://python.langchain.com/docs/how_to/custom_tools/
@tool
def get_holidays(country: str, year: int = None) -> str:
    """
    Get holidays with params of country and year through the Calendarific API.
    Input is country, year e.g., 'US, 2024'.
    Current year is applied if year is not specified.
    """

    # Spliting the country and year
    if "," in country:
        country, year_str = country.split(",")
        try:
            year = int(year_str)
        except ValueError:
            return "Error: Invalid year format. Please use 'country_code,year' (e.g., 'US,2024')."
    # Bounding check for the year
    if year is None:
        year = datetime.datetime.now().year
    if year > 2049:
        return "Error: Calendarific API only supports years up to 2049."

    # Query holidays by the given country and year through Calendarific API
    try:
        params = {
            "api_key": CALENDARIFIC_API_KEY,
            "country": country,
            "year": year,
        }
        response = requests.get(CALENDARIFIC_API_URL, params = params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json() # Parse JSON response
		
		# Parse the result according to the API documentation available at
		# https://calendarific.com/api-documentation
        if data["meta"]["code"] == 200: # Retrieve the result code, 200: API call success
            holidays = data["response"]["holidays"] # Retrieve the holidays
            if not holidays:
                return f"No holidays found for {country} in {year}."

            # Format each holiday as "- name (date)" on a separate line
            return "\n".join([f"- {h['name']} ({h['date']['iso']})" for h in holidays])
        else:
            return f"API Error: {data['meta']['code']} - {data['meta']['error_type']}"
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Calendarific API: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing Calendarific API response: {e}. \
            Raw Response: {response.text if 'response' in locals() else 'No response received'}"

def get_model():
    model = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    return model

def generate_hw01(question):
    system_prompt_str = """
        You are the assistant to provide holiday relevant query.
        For the query result, please porvide dates and names info in exactly structured JSON format as follows.
X        {{
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

    model = get_model()
    response = model.invoke(messages)
    result = response.content.strip().removeprefix("```json").removesuffix("```")
    return result
    
def generate_hw02(question):
    system_prompt_str = """
        You are the assistant to provide holiday relevant query through the tool `get_holidays`.
        For the query result, please porvide dates and names info in exactly structured JSON format as follows.
        The language of the output result is same as the input.
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}
    """

    # the prompt is from https://smith.langchain.com/hub/hwchase17/openai-functions-agent
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt.messages)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str),
            ("placeholder", "{chat_history}"), # MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html
    tools = [
        Tool(
            name = get_holidays.name,
            func = get_holidays,
            description = get_holidays.description,
            parameters = [
                ("country", str),
                ("year", int),
            ]
        )
    ]

    model = get_model()
    agent = create_tool_calling_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools)
    response = agent_executor.invoke({"input": question})
    result = response["output"].strip().removeprefix("```json").removesuffix("```")
    return result
    
def generate_hw03(question2, question3):
    system_prompt_str = """
        You are the assistant to provide holiday relevant query through the tool `get_holidays`.
        For the query result, please porvide dates and names info in exactly structured JSON format as follows.
        The language of the output result is same as the input.
        {{
            "Result": [
                {{
                    "date": "2024-12-25",
                    "name": "Christmas"
                }}
            ]
        }}

        If the question is like "Is the given holiday in the list?",
        you should check the date AND name in the query result list to check if the holiday exists.
        It's one new holiday if either of date and same name is different.
        For the new holiday, please hint user in exactly structured JSON format as follows.
        The language of the output result is same as the input.
        {{
            "Result": {{
                "add": true/false,
                "reason": "Describe why you do or do not want to add a new holiday, 
                        specify whether the holiday already exists in the list, 
                        and shows all the contents of the current list for reference."
            }}
        }}
    """

    # the prompt is from https://smith.langchain.com/hub/hwchase17/openai-functions-agent
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt.messages)
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str),
            ("placeholder", "{chat_history}"), # MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html
    tools = [
        Tool(
            name = get_holidays.name,
            func = get_holidays,
            description = get_holidays.description,
            parameters = [
                ("country", str),
                ("year", int),
            ]
        )
    ]

    model = get_model()
    agent = create_tool_calling_agent(model, tools, agent_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools)

    # Create chat history
    histories = {}
    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in histories:
            histories[session_id] = ChatMessageHistory()
        return histories[session_id]

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
    )

    session_id = "query_holiday"
    agent_with_chat_history.invoke(
        {"input": question2},
        config = {"configurable": {"session_id": session_id}},
    )
    response = agent_with_chat_history.invoke(
        {"input": question3},
        config = {"configurable": {"session_id": session_id}},
    )
    result = response["output"].strip().removeprefix("```json").removesuffix("```")
    return result
    
def generate_hw04(question):
    def encode_to_image_url(image_path, image_type = "jpeg"):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/{image_type};base64,{encoded_string}"  # Correct MIME type
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None

    system_prompt_str = """
        You are the assistant to query game score.
        For the query result, please porvide score info in exactly structured JSON format as follows.
        {{
            "Result": {{
                "score": 1234
            }}
        }}
        The language of the output result is same as the input.
    """
    base64_image = encode_to_image_url("baseball.png")
    messages = [
        SystemMessage(content = [{"type": "text", "text": system_prompt_str},]),
        HumanMessage(content = [
            {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            ],
        ),
    ]

    model = get_model()
    response = model.invoke(messages)
    result = response.content.strip().removeprefix("```json").removesuffix("```")
    return result
    
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

# print("=== hw01 output ===")
# print(generate_hw01("2024年台灣10月紀念日有哪些?"))

# print("=== hw02 output ===")
# print(generate_hw02("2024年台灣10月紀念日有哪些?"))

# print("=== hw03 output ===")
# print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'))

print("=== hw04 output ===")
print(generate_hw04('請問中華台北的積分是多少'))
