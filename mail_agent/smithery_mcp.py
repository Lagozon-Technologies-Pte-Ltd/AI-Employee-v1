import json
import base64
import mcp
from mcp.client.streamable_http import streamablehttp_client

smithery_api_key = "f0ec3b8b-973e-4bc0-95f5-433a53d09e6e"
smithery_profile = "developing-ladybug-IOtpIS"

url = f"https://server.smithery.ai/@shinzo-labs/gmail-mcp/mcp?api_key={smithery_api_key}&profile={smithery_profile}"

async def process_emails_and_create_drafts():
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Step 1: List unread messages
            list_result = await session.call_tool(
                "list_messages",
                arguments={"maxResults": 5, "q": "is:unread"}
            )
            raw_json = list_result.content[0].text
            list_json = json.loads(raw_json)
            message_ids = [msg["id"] for msg in list_json.get("messages", [])]

            created_drafts = []

            # Step 2: Process each message
            for msg_id in message_ids:
                msg_detail = await session.call_tool("get_message", arguments={"id": msg_id})
                msg_data = json.loads(msg_detail.content[0].text)

                headers = msg_data.get("payload", {}).get("headers", [])
                subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
                sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown Sender")
                thread_id = msg_data.get("threadId")

                # Step 3: AI-generated draft (placeholder)
                ai_reply = f"Hello, this is an automated draft reply to '{subject}'.\n\nBest Regards,\nYour AI Assistant"

                # Step 4: Create MIME for Gmail
                raw_email = f"From: me\nTo: {sender}\nSubject: Re: {subject}\n\n{ai_reply}"
                raw_base64 = base64.urlsafe_b64encode(raw_email.encode("utf-8")).decode("utf-8")

                # Step 5: Create draft
                draft_result = await session.call_tool(
                    "create_draft",
                    arguments={
                        "message": {
                            "threadId": thread_id,
                            "raw": raw_base64
                        }
                    }
                )
                created_drafts.append({"subject": subject, "to": sender})

            return created_drafts
