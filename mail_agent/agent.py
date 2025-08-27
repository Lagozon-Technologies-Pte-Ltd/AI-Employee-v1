import os
import json, sys, time, mcp,re
import yaml
import logging
import base64
from openai import OpenAI
from config import openai_api_key
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

client = OpenAI(api_key=openai_api_key)
base_dir = os.path.dirname(__file__)
# Load LLM prompts from YAML
def load_prompts():
    with open(os.path.join(base_dir, "prompts.yaml"), "r") as f:
        return yaml.safe_load(f)

PROMPTS = load_prompts()

# Load MCP config.json and tool schemas
with open("config.json", "r") as f:
    CONFIG = json.load(f)
with open(os.path.join(base_dir, "tools_schema.json"), "r") as f:
    TOOL_FORMATS = json.load(f)


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
                    tool_args = prepare_tool_args(tool_name,tool_schema, tool_args, context_store)

                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        parsed_result = json.loads(result.content[0].text)

                        # Update context store with returned data
                        # Update context store with returned data
                        # Update context store with returned data
                        if isinstance(parsed_result, dict):
                            # Store message IDs with meaningful keys
                            if tool_name == "list_messages" and "messages" in parsed_result:
                                messages = parsed_result.get("messages", [])
                                for i, message in enumerate(messages):
                                    context_store[f"message_{i+1}_id"] = message.get("id")
                            
                            # Store individual message data and extract headers
                            elif tool_name == "get_message" and "id" in parsed_result:
                                message_id = parsed_result.get("id")
                                context_store[f"message_data"] = parsed_result
                                context_store[f"message_{message_id}_data"] = parsed_result  # Also store by ID

                                # Extract headers (lowercase keys)
                                if 'payload' in parsed_result and 'headers' in parsed_result['payload']:
                                    headers = parsed_result['payload']['headers']
                                    header_dict = {header['name'].lower(): header['value'] for header in headers}

                                    # Store headers with consistent naming
                                    context_store["message_from"] = header_dict.get('from', '')
                                    context_store["message_to"] = header_dict.get('to', '')
                                    context_store["message_subject"] = header_dict.get('subject', '')
                                
                                # Store snippet (plain text short summary provided by Gmail)
                                context_store["snippet"] = parsed_result.get("snippet", "")
                                

                            # Always update context with the full result
                            context_store.update(parsed_result)                        # Add successful result to responses
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

def generate_reply_subject_body(message_from: str, message_subject: str, snippet: str):
    """
    Ask the LLM to compose a reply subject and body based on original message headers/snippet.
    Returns a dict: {"subject": "...", "body": "..."}.
    Falls back to simple defaults on error.
    """
    try:
        # Compose the instruction for the LLM
        instruction = (
            "You are a professional email assistant. Read the original message below and "
            "compose a reply. Output ONLY valid JSON with exactly two keys: "
            '"subject" and "body". Do not add any explanation or extra text.\n\n'
            "Original message headers:\n"
            f"From: {message_from}\n"
            f"Subject: {message_subject}\n\n"
            "Message snippet:\n"
            f"{snippet}\n\n"
            "Requirements:\n"
            "- Subject: short, appropriate, prefer 'Re: <original>' unless a clearer subject is better.\n"
            "- Body: plain text, polite, concise (approx 2-5 short sentences). Sign-off optional.\n"
            "- Return JSON only, e.g. {\"subject\": \"...\", \"body\": \"...\"}.\n"
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that composes email replies in JSON."},
                {"role": "user", "content": instruction}
            ],
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()

        # Extract the first JSON object found in the LLM output
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError("No JSON found in LLM output")

        parsed = json.loads(m.group(0))
        subject = parsed.get("subject", "").strip()
        body = parsed.get("body", "").strip()

        # Safety / fallback: ensure non-empty
        if not subject:
            subject = f"Re: {message_subject}" if message_subject else "Re:"
        if not body:
            body = "Thanks for your message. I'll get back to you shortly."

        return {"subject": subject, "body": body}
    except Exception as e:
        logger.error(f"generate_reply_subject_body failed: {e}")
        # fallback defaults
        fallback_subject = f"Re: {message_subject}" if message_subject else "Re:"
        fallback_body = "Thanks for your message. I'll get back to you shortly."
        return {"subject": fallback_subject, "body": fallback_body}

def resolve_placeholders(value, context_store):
    """Recursively resolve placeholders from context_store in strings, lists, and dicts."""
    if isinstance(value, str):
        # Handle complex {{placeholder}} patterns with regex
        import re
        placeholders = re.findall(r'{{(.*?)}}', value)
        if placeholders:
            resolved_value = value
            for placeholder in placeholders:
                placeholder = placeholder.strip()
                # Try exact match first
                if placeholder in context_store:
                    resolved_value = resolved_value.replace(f'{{{{{placeholder}}}}}', str(context_store[placeholder]))
                else:
                    # Try to find the best match
                    found = False
                    for key in context_store.keys():
                        if placeholder.lower() in key.lower() or key.lower() in placeholder.lower():
                            resolved_value = resolved_value.replace(f'{{{{{placeholder}}}}}', str(context_store[key]))
                            found = True
                            break
                    if not found:
                        # If not found, keep the placeholder
                        logger.warning(f"Placeholder {placeholder} not found in context")
            return resolved_value
        # Handle direct context keys
        elif value in context_store:
            return context_store[value]
        return value
    elif isinstance(value, list):
        return [resolve_placeholders(v, context_store) for v in value]
    elif isinstance(value, dict):
        return {k: resolve_placeholders(v, context_store) for k, v in value.items()}
    return value
def prepare_tool_args(tool_name, tool_schema, user_args, context_store):
    """
    Auto-populates missing required args using values from context_store.
    For create_draft, subject & body are generated by LLM (based on snippet).
    """
    schema_args = tool_schema.get("parameters", {}).get("properties", {})
    required_args = tool_schema.get("parameters", {}).get("required", [])

    # First resolve any placeholders
    final_args = resolve_placeholders(user_args.copy(), context_store)

    # Special handling for create_draft
    if tool_name == "create_draft":
        # If subject/body already provided by LLM plan, just keep them
        if "subject" in final_args and "body" in final_args:
            return final_args  

        # Otherwise, try to generate based on context (for replies)
        message_data = context_store.get("message_data", {})
        snippet = context_store.get("snippet", "")

        if message_data:  # <-- Only for replies
            from_address, subject = "", ""
            if 'payload' in message_data and 'headers' in message_data['payload']:
                headers = message_data['payload']['headers']
                header_dict = {header['name'].lower(): header['value'] for header in headers}
                from_address = header_dict.get('from', '')
                subject = header_dict.get('subject', '')

            reply = generate_reply_subject_body(from_address, subject, snippet)
            final_args["subject"] = reply["subject"]
            final_args["body"] = reply["body"]

    else:
        # Auto-fill required args for other tools
        for arg_name in required_args:
            if arg_name not in final_args:
                for key in context_store.keys():
                    if key.lower() == arg_name.lower() or key.endswith(f'_{arg_name}'):
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