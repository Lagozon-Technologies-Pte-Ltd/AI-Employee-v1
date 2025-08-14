from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from agent import handle_user_query

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})

@app.post("/", response_class=HTMLResponse)
async def query(request: Request, user_input: str = Form(...)):
    print("POST request received with input:", user_input)
    result = await handle_user_query(user_input)
    print("Got result:", result)
    
    # TEST: Return a simple HTML response (bypass Jinja2)
    return HTMLResponse(content=f"""
        <html>
            <body>
                <h1>Test Response</h1>
                <pre>{result}</pre>
            </body>
        </html>
    """)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
