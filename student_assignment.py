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
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
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

print("=== hw01 output ===")
print(generate_hw01("2024年台灣10月紀念日有哪些?"))

print("=== hw02 output ===")
print(generate_hw02("2024年台灣10月紀念日有哪些?"))
