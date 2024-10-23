#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: flask.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""

import os
import traceback
from flask import Flask, Response, request, jsonify
from .llm import LLMApi
from . import sender, data, auth

app = Flask("llm-proxy")

def convert_resonse(resp: sender.Response) -> Response:
    return Response(resp.text, resp.status_code, resp.headers)

def resp(code: data.ErrMsg, msg: str, *args, **kwargs) -> Response:
    return jsonify(data.resp(code, msg, *args, **kwargs))

@app.errorhandler(Exception)
def handle_error(error):
    traceback.print_exc()
    return resp(data.ErrMsg.UNKNOWN_ERROR, "Exception occurred", error=str(error)), 500

@app.before_request
def check_token():
    auth_header = request.headers.get('Authorization')
    token = data.get_bearer_token(auth_header)
    
    if auth.verify_token(token, data.secret) is False:
        return resp(data.ErrMsg.AUTH_ERROR, "Authorization failed"), 401

@app.route("/v1/chat/completions", methods=['POST'])
def chat():
    return convert_resonse(LLMApi().chat(request.get_json(force=True)))

@app.route("/v1/models", methods=['POST'])
def models():
    return convert_resonse(LLMApi().models())

@app.route("/gen_token")
def gen_token():
    secret = request.args.get('secret', '')
    token = auth.gen_token(secret)

    return resp(data.ErrMsg.OK, "操作成功", token=token)


def run_app(host: str, port: int) -> Flask:
    flask_env = os.environ.get('FLASK_ENV')
    if flask_env != 'production':
        app.config['ENV'] = 'development'
        app.config['DEBUG'] = True

    if port > 0:
        app.run(host=host, port=port)
    return app
