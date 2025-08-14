import json, sys, time, mcp
import yaml
import logging
from openai import OpenAI
from config import openai_api_key
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

client = OpenAI(api_key=openai_api_key)

# Load LLM prompts from YAML
def load_prompts():
    with open("prompts.yaml", "r") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# Load MCP config.json
with open("config.json", "r") as f:
    CONFIG = json.load(f)

async def handle_user_query(user_input: str):
    gmail_server_cfg = CONFIG["mcpServers"]["gmail"]
    server_params = StdioServerParameters(
        command=gmail_server_cfg["command"],
        args=gmail_server_cfg["args"],
    )
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_info = await session.list_tools()
                tool_names = [tool.name for tool in tools_info.tools]
                logger.info(f"Available tools: {tool_names}")

                tool_name, tool_args = choose_tool_with_llm(user_input, tool_names)
                logger.info(f"Chosen tool: {tool_name} with args: {tool_args}")

                if not tool_name:
                    return {"error": "No suitable tool found."}

                start = time.time()
                result = await session.call_tool(tool_name, arguments=tool_args)
                logger.info(f"Time after call_tool: {time.time() - start:.2f}s")

                parsed_result = json.loads(result.content[0].text)
                logger.info(f"parsed result: {parsed_result}")

                if tool_name == "list_messages" and "messages" in parsed_result:
                    detailed_messages = []
                    for msg in parsed_result["messages"]:
                        msg_id = msg.get("id")
                        if msg_id:
                            try:
                                logger.info(f"Fetching message {msg_id}")
                                msg_result = await session.call_tool(
                                    "get_message",
                                    arguments={"id": msg_id}
                                )
                                logger.info(f"Message {msg_id} fetched")
                                detailed_messages.append(
                                    json.loads(msg_result.content[0].text)
                                )
                            except Exception as e:
                                logger.error(f"Failed to fetch message {msg_id}: {e}")
                    
                    return {
                        "tool": "list_messages + get_message",
                        "result": detailed_messages
                    }

                logger.info(f"tool: {tool_name}, parsed result: {parsed_result} just before returning!")
                return {"tool": tool_name, "result": parsed_result}

    except Exception as e:
        logger.error(f"Error in MCP session: {e}")
        return {"error": str(e)}
    finally:
        logger.info("Session closed successfully")
def choose_tool_with_llm(user_input: str, tool_names: list):
    prompt = PROMPTS["select_tool"].format(
        tool_list=tool_names, user_input=user_input
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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
