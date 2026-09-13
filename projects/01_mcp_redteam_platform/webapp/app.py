"""The real chat UI: FastAPI + Jinja2, backed by one long-lived MCPAgentSession
(hardened=True — production always runs hardened; the vulnerable configs only
ever run inside redteam/ measurement scripts)."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.client import MCPAgentSession  # noqa: E402

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
chat_history: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with MCPAgentSession(hardened=True) as session:
        app.state.session = session
        yield


app = FastAPI(title="mcp-lab red-team platform", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "chat.html", {"history": chat_history})


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)) -> HTMLResponse:
    result = await request.app.state.session.run_turn(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append(
        {"role": "assistant", "content": result["answer"], "trace": result["trace"]}
    )
    return templates.TemplateResponse(request, "chat.html", {"history": chat_history})
