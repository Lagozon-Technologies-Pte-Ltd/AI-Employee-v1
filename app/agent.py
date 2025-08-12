import json, mcp, yaml
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI
import logging
from logging import getLogger
from config import openai_api_key, smithery_api_key, smithery_profile
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(api_key=openai_api_key)

url = f"https://server.smithery.ai/@shinzo-labs/gmail-mcp/mcp?api_key={smithery_api_key}&profile={smithery_profile}"
def load_prompts():
    with open("prompts.yaml", "r") as f:
        return yaml.safe_load(f)
PROMPTS = load_prompts()

async def handle_user_query(user_input: str):
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Step 1: Get available tools
            tools_info = await session.list_tools()
            tool_names = [tool.name for tool in tools_info.tools]

            # Step 2: Ask LLM which tool to use
            tool_name, tool_args = choose_tool_with_llm(user_input, tool_names)
            logger.info(f"Chosen tool: {tool_name} with args: {tool_args}")

            if not tool_name:
                return {"error": "No suitable tool found."}

            # Step 3: Call the main tool
            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                parsed_result = json.loads(result.content[0].text)

                # Step 4: If list_messages, chain into get_message for each ID
                if tool_name == "list_messages" and "messages" in parsed_result:
                    detailed_messages = []
                    for msg in parsed_result["messages"]:
                        msg_id = msg.get("id")
                        if msg_id:
                            try:
                                msg_result = await session.call_tool(
                                    "get_message",
                                    arguments={"id": msg_id}
                                )
                                detailed_messages.append(json.loads(msg_result.content[0].text))
                            except Exception as e:
                                logger.error(f"Failed to fetch message {msg_id}: {e}")
                    
                    return {
                        "tool": "list_messages + get_message",
                        "result": detailed_messages
                    }

                return {
                    "tool": tool_name,
                    "result": parsed_result
                }

            except Exception as e:
                return {"error": str(e)}
def choose_tool_with_llm(user_input: str, tool_names: list):
    """
    Uses OpenAI to pick the right tool and arguments from a list.
    """
    prompt = PROMPTS["select_tool"].format(tool_list=tool_names, user_input=user_input)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Select the correct tool and arguments in JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        parsed = json.loads(response.choices[0].message.content.strip())
        logger.info(f"LLM response: {parsed}")
        return parsed["tool"], parsed.get("arguments", {})
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        raise ValueError(f"Invalid JSON from LLM: {e}")
