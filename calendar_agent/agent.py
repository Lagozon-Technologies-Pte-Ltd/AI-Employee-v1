import json, time, mcp, re, os
import yaml
import logging
from openai import OpenAI
from config import openai_api_key
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

client = OpenAI(api_key=openai_api_key)
base_dir = os.path.dirname(__file__)

# -------------------------
# Load Prompts
# -------------------------
def load_prompts():
    with open(os.path.join(base_dir, "prompts.yaml"), "r") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# -------------------------
# Load Config + Tool Schemas
# -------------------------
with open("config.json", "r") as f:
    CONFIG = json.load(f)
with open(os.path.join(base_dir, "tools_schema.json"), "r") as f:
    TOOL_FORMATS = json.load(f)

# -------------------------
# Helper: Prepare Tool Args
# -------------------------
def prepare_tool_args(tool_name, tool_schema, provided_args, context_store):
    args = {}
    for arg_name, arg_schema in tool_schema.get("arguments", {}).items():
        if arg_name in provided_args:
            args[arg_name] = provided_args[arg_name]
        elif arg_name in context_store:
            args[arg_name] = context_store[arg_name]
        else:
            args[arg_name] = arg_schema.get("default", None)
    return args

# -------------------------
# Helper: Choose tool with LLM
# -------------------------
def choose_tool_with_llm(user_input):
    prompt = PROMPTS.get("select_tool", "")
    tool_usage = json.dumps(TOOL_FORMATS, indent=2)

    formatted_prompt = prompt.format(
        tool_usage=tool_usage,
        user_input=user_input
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": formatted_prompt}],
        temperature=0,
    )

    try:
        content = response.choices[0].message.content.strip()
        plan_json = json.loads(content)
        return plan_json.get("plan", [])
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return []

# -------------------------
# Helper: Generate response for user
# -------------------------
def generate_response(user_input: str, tool_results: list):
    prompt = PROMPTS["response_generator"].format(
        user_input=user_input,
        tool_response=json.dumps(tool_results, indent=2)
    )
    logger.info(f"Response generation prompt: {prompt}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    try:
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error extracting LLM response: {e}")
        raise ValueError(f"Failed to extract response from LLM: {e}")

# -------------------------
# Context Store Management for Calendar
# -------------------------
def update_context_from_calendar_result(context_store, tool_name, parsed_result):
    """Update context store with calendar-specific data"""
    if not isinstance(parsed_result, dict):
        return
    
    # Store calendar IDs from list-calendars
    if tool_name == "list-calendars" and "calendars" in parsed_result:
        calendars = parsed_result.get("calendars", [])
        for i, calendar in enumerate(calendars):
            context_store[f"calendar_{i+1}_id"] = calendar.get("id")
            context_store[f"calendar_{i+1}_name"] = calendar.get("summary", "")
    
    # Store event data from list-events or get-event
    elif tool_name in ["list-events", "get-event"]:
        if "events" in parsed_result:
            events = parsed_result.get("events", [])
            for i, event in enumerate(events):
                context_store[f"event_{i+1}_id"] = event.get("id")
                context_store[f"event_{i+1}_summary"] = event.get("summary", "")
        elif "id" in parsed_result:
            event_id = parsed_result.get("id")
            context_store[f"event_data"] = parsed_result
            context_store[f"event_{event_id}_data"] = parsed_result
    
    # Store primary calendar ID if available
    if "primary" in parsed_result:
        context_store["primary_calendar_id"] = parsed_result.get("primary")
    
    # Always update context with the full result
    context_store.update(parsed_result)

# -------------------------
# Main: Generic Handler
# -------------------------
import traceback

# Update your main handler to capture more detailed errors
async def handle_user_query(user_input: str):
    server_params = StdioServerParameters(
        command="node",
        env={
            "GOOGLE_OAUTH_CREDENTIALS": "C:\\Users\\dell\\Documents\\GitHub\\AI-Employee-v1\\calendar_agent\\gcp-oauth.keys.json"
        },
        args=["C:/Users/dell/Documents/GitHub/google-calendar-mcp/build/index.js"],
    )

    print("Launching MCP with:", server_params)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("MCP session initialized successfully")
                
                # Get the plan (multiple steps) from LLM
                plan = choose_tool_with_llm(user_input)
                logger.info(f"Generated plan: {plan}")

                if not plan or not isinstance(plan, list):
                    return {"error": "No suitable plan generated."}

                results = []
                context_store = {}
                start = time.time()

                for step in plan:
                    tool_name = step.get("tool")
                    tool_args = step.get("arguments", {})
                    if not tool_name:
                        continue

                    tool_schema = TOOL_FORMATS.get(tool_name, {})
                    tool_args = prepare_tool_args(tool_name, tool_schema, tool_args, context_store)

                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        parsed_result = json.loads(result.content[0].text)

                        update_context_from_calendar_result(context_store, tool_name, parsed_result)
                        
                        results.append({
                            "tool": tool_name,
                            "response": parsed_result
                        })

                    except Exception as e:
                        logger.error(f"Failed to execute {tool_name}: {e}")
                        logger.error(traceback.format_exc())
                        results.append({
                            "tool": tool_name,
                            "error": str(e)
                        })

                logger.info(f"Execution completed in {time.time() - start:.2f}s")
                generated_response = generate_response(user_input, results)
                logger.info(f"Generated response: {generated_response}")

                return {
                    "plan": plan,
                    "tool_responses": results,
                    "bot_response": generated_response
                }

    except Exception as e:
        logger.error(f"Error in MCP session: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        logger.info("Session closed successfully")