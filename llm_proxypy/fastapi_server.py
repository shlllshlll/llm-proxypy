#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: fastapi.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""

import logging
import traceback
from typing import Awaitable
from fastapi import FastAPI, Request, status
from fastapi.responses import StreamingResponse, Response, JSONResponse
import uvicorn
from .llm import LLMApi
from . import sender, data, auth


app = FastAPI()

def convert_resonse(resp: sender.Response, stream: bool = False) -> Response:
    if stream:
        return StreamingResponse(resp.text, resp.status_code, resp.headers)
    else:
        return Response(resp.text, resp.status_code, resp.headers)

def resp(code: data.ErrMsg, msg: str, status_code: int, *args, **kwargs) -> JSONResponse:
    return JSONResponse(content=data.resp(code, msg, *args, **kwargs), status_code=status_code)

@app.middleware("http")
async def check_token(request: Request, call_next: Awaitable[Request]):
    auth_header = request.headers.get('Authorization')
    token = data.get_bearer_token(auth_header)

    if auth.verify_token(token, data.secret) is False:
        return resp(data.ErrMsg.AUTH_ERROR, "Authorization header missing or incorrect", status.HTTP_401_UNAUTHORIZED)
    response = await call_next(request)
    return response

@app.exception_handler(Exception)
async def handle_error(request: Request, exc: Exception):
    traceback.print_exc()
    return resp(data.ErrMsg.UNKNOWN_ERROR, "Exception occurred", status.HTTP_500_INTERNAL_SERVER_ERROR, error=str(exc))

@app.route("/v1/chat/completions", methods=['POST'])
async def chat(request: Request):
    request_body = await request.json()
    is_stream = request_body.get("stream", False)
    return convert_resonse(await LLMApi().async_chat(request_body), is_stream)

@app.route("/v1/models", methods=['POST'])
async def models(request: Request):
    nextchat = request.query_params.get("nextchat", 0)
    return convert_resonse(await LLMApi().async_models(nextchat))

@app.route("/gen_token")
def gen_token(secret: str = ''):
    token = auth.gen_token(secret)

    return resp(data.ErrMsg.OK, "操作成功", token=token)

def run_app(host: str, port: int) -> FastAPI:
    if port > 0:
        uvicorn.run("llm_proxypy.fastapi_server:app", host=host, port=port, reload=True)
    return app
