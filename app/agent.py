import json, sys, time, mcp
import yaml
import logging
import base64
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

# Load MCP config.json and tool schemas
with open("config.json", "r") as f:
    CONFIG = json.load(f)
with open("tools_schema.json", "r") as f:
    TOOL_FORMATS = json.load(f)

def prepare_email_content(message_data):
    """Convert message data to proper format for create_draft"""
    if not isinstance(message_data, dict):
        return None
    
    # Extract relevant parts from the message data
    subject = message_data.get('subject', 'No Subject')
    body = message_data.get('body', '')
    from_email = message_data.get('from', '')
    to_email = message_data.get('to', '')
    
    # Create a simple email format (this is simplified)
    email_content = f"""From: {from_email}
To: {to_email}
Subject: {subject}

{body}"""
    
    # Base64 encode
    return base64.urlsafe_b64encode(email_content.encode()).decode()

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

                # Get the plan (multiple steps) from LLM
                plan = choose_tool_with_llm(user_input, tool_names)
                logger.info(f"Generated plan: {plan}")

                if not plan or not isinstance(plan, list):
                    return {"error": "No suitable plan generated."}

                results = []
                context_store = {}  # Store key-value pairs from previous tool outputs
                start = time.time()

                for step in plan:
                    tool_name = step.get("tool")
                    tool_args = step.get("arguments", {})
                    if not tool_name:
                        continue

                    # Auto-fill missing args from context using schema
                    tool_schema = TOOL_FORMATS.get(tool_name, {})
                    tool_args = prepare_tool_args(tool_schema, tool_args, context_store)

                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        parsed_result = json.loads(result.content[0].text)

                        # Update context store with returned data
                        if isinstance(parsed_result, dict):
                            # Store message IDs with meaningful keys
                            if tool_name == "list_messages" and "messages" in parsed_result:
                                for i, message in enumerate(parsed_result.get("messages", [])):
                                    context_store[f"message_{i+1}_id"] = message.get("id")
                            
                            # Store individual message data and extract headers
                            elif tool_name == "get_message" and "id" in parsed_result:
                                message_id = parsed_result.get("id")
                                context_store[f"message_{message_id}_data"] = parsed_result
                                
                                # Extract and store headers with consistent naming
                                if 'payload' in parsed_result and 'headers' in parsed_result['payload']:
                                    headers = parsed_result['payload']['headers']
                                    header_dict = {header['name'].lower(): header['value'] for header in headers}
                                    
                                    # Store individual headers for easy access
                                    context_store[f"message_{message_id}_from"] = header_dict.get('from', '')
                                    context_store[f"message_{message_id}_to"] = header_dict.get('to', '')
                                    context_store[f"message_{message_id}_subject"] = header_dict.get('subject', '')
                                    
                                    # Also store the full header dict for complex access
                                    context_store[f"message_{message_id}_headers"] = header_dict
                                
                                # Store snippet
                                if 'snippet' in parsed_result:
                                    context_store[f"message_{message_id}_snippet"] = parsed_result.get('snippet', '')
                            
                            context_store.update(parsed_result)

                        # Add successful result to responses
                        results.append({
                            "tool": tool_name,
                            "response": parsed_result
                        })

                    except Exception as e:
                        logger.error(f"Failed to execute {tool_name}: {e}")
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
        return {"error": str(e)}
    finally:
        logger.info("Session closed successfully")


def resolve_placeholders(value, context_store):
    """Recursively resolve placeholders from context_store in strings, lists, and dicts."""
    if isinstance(value, str):
        # Handle {{placeholder}} format
        if '{{' in value and '}}' in value:
            start = value.find('{{')
            end = value.find('}}')
            if start != -1 and end != -1 and end > start:
                placeholder = value[start+2:end].strip()
                resolved_value = context_store.get(placeholder, value)
                # Replace only the placeholder part, keep surrounding text
                return value[:start] + str(resolved_value) + value[end+2:]
        # Handle direct context keys
        elif value in context_store:
            return context_store[value]
        return value
    elif isinstance(value, list):
        return [resolve_placeholders(v, context_store) for v in value]
    elif isinstance(value, dict):
        return {k: resolve_placeholders(v, context_store) for k, v in value.items()}
    return value


def prepare_tool_args(tool_schema, user_args, context_store):
    """
    Auto-populates missing required args using values from context_store.
    Enhanced to handle email header extraction and better placeholder resolution.
    """
    # First pass: resolve all placeholders recursively
    final_args = resolve_placeholders(user_args.copy(), context_store)
    
    # Special handling for email-related tools
    for arg_name, arg_value in final_args.items():
        if isinstance(arg_value, str):
            # Handle message_id fallback
            if arg_value.endswith('_id') and arg_value in context_store:
                final_args[arg_name] = context_store[arg_value]
            
            # Handle email header placeholders that weren't resolved
            elif '{{' in arg_value and '}}' in arg_value:
                # Extract placeholder name
                placeholder = arg_value[arg_value.find('{{')+2:arg_value.find('}}')].strip()
                
                # Try to find the value in context store
                if placeholder in context_store:
                    final_args[arg_name] = context_store[placeholder]
                else:
                    # Look for message-specific headers
                    for key in context_store.keys():
                        if key.startswith('message_') and placeholder in key:
                            final_args[arg_name] = context_store[key]
                            break

    # Additional fallback for common email fields
    if 'to' in final_args and isinstance(final_args['to'], str) and '{{' in final_args['to']:
        # Try to extract from address from message data
        for key in context_store.keys():
            if key.endswith('_from') and key.startswith('message_'):
                final_args['to'] = context_store[key]
                break

    if 'subject' in final_args and isinstance(final_args['subject'], str) and '{{' in final_args['subject']:
        # Try to extract subject from message data
        for key in context_store.keys():
            if key.endswith('_subject') and key.startswith('message_'):
                final_args['subject'] = context_store[key]
                break

    # Auto-fill required args if still missing
    for arg_name, arg_props in schema_args.items():
        if arg_name not in final_args and arg_props.get("required"):
            if arg_name in context_store:
                final_args[arg_name] = context_store[arg_name]
            else:
                for key in context_store.keys():
                    if key.lower() == arg_name.lower():
                        final_args[arg_name] = context_store[key]
                        break

    logger.info(f"Prepared tool args after replacement: {final_args}")
    return final_args

def choose_tool_with_llm(user_input: str, tool_names: list):
    try:
        prompt = PROMPTS["select_tool"].format(
            tool_list=tool_names, user_input=user_input, tool_usage=TOOL_FORMATS
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Select the correct tool(s) and arguments in JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        try:
            raw_output = response.choices[0].message.content.strip()
            parsed = json.loads(raw_output)
            return parsed.get("plan", [])
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            raise ValueError(f"Invalid JSON from LLM: {e}")
    except Exception as e:
        logger.error(f"Error in LLM tool selection: {e}")
        raise


def generate_response(user_input: str, tool_results: list):
    prompt = PROMPTS["response_generator"].format(
        user_input=user_input,
        tool_response=json.dumps(tool_results, indent=2)
    )
    logger.info(f"Response generation prompt: {prompt}")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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