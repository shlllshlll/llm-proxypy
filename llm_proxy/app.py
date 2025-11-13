#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: app.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""

from . import fastapi_server
from .settings import ServerSettings
from .config import init_llm

settings = ServerSettings()    # type: ignore[reportCallIssue]
fastapi_server.app.state.settings = settings
init_llm()
app = fastapi_server.app