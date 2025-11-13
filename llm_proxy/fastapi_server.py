#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: fastapi.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""

from contextlib import asynccontextmanager
import traceback
import logging
from fastapi import APIRouter, Depends, FastAPI, Request, status, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
import uvicorn
from .llm import LLMApi
from . import sender, data, auth
from .config import init_logging
from .settings import ServerSettings

logger = logging.getLogger(__name__)
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: ServerSettings = app.state.settings
    init_logging(settings)

    yield


def convert_resonse(resp: sender.Response, stream: bool = False) -> Response:
    if stream:
        return StreamingResponse(resp.text, resp.status_code, resp.headers)
    else:
        return Response(resp.text, resp.status_code, resp.headers)


def resp(
    code: data.ErrMsg, msg: str, status_code: int, *args, **kwargs
) -> JSONResponse:
    return JSONResponse(
        content=data.resp(code, msg, *args, **kwargs), status_code=status_code
    )

async def check_auth(request: Request):
    auth_header = request.headers.get("Authorization", "")
    token = data.get_bearer_token(auth_header)
    if auth.verify_token(token, LLMApi().secret) is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization failed"
        )

api_no_auth = APIRouter(prefix="", dependencies=[])
api_auth = APIRouter(prefix="", dependencies=[Depends(check_auth)])

@app.exception_handler(Exception)
async def handle_error(request: Request, exc: Exception):
    traceback.print_exc()
    return resp(
        data.ErrMsg.UNKNOWN_ERROR,
        "Exception occurred",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        error=traceback.format_exception_only(type(exc), exc),
    )


@api_auth.post("/v1/chat/completions")
async def chat(request: Request):
    request_body = await request.json()
    is_stream = request_body.get("stream", False)
    return convert_resonse(await LLMApi().async_chat(request_body), is_stream)


@api_no_auth.get("/v1/models")
async def models(request: Request):
    return convert_resonse(await LLMApi().async_models())


@api_no_auth.get("/get_token")
def get_token(secret: str = ""):
    print("Generating token with secret:", secret)
    token = auth.gen_token(secret)

    return resp(data.ErrMsg.OK, "操作成功", token=token, status_code=status.HTTP_200_OK)


def run_app(host: str, port: int, reload: bool) -> FastAPI:
    if port > 0:
        uvicorn.run("llm_proxy.main:app", host=host, port=port, reload=reload)
    return app

app.include_router(api_no_auth)
app.include_router(api_auth)
