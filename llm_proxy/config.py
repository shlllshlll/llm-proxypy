#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: config.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""
import logging
import sys
import os
import yaml
from .settings import ServerSettings
from . import fastapi_server, llm

def init_logging(settings: ServerSettings):
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s] - %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    root_logger.addHandler(stream_handler)

    logging.getLogger("uvicorn.access").handlers = root_logger.handlers
    logging.getLogger("uvicorn.error").handlers = root_logger.handlers

    logger = logging.getLogger(__name__)
    logger.info(f"Logger initialized with level: {settings.LOG_LEVEL} in PID {os.getpid()}")

def init_llm():
    settings: ServerSettings = fastapi_server.app.state.settings
    with open(settings.CONF, 'r') as f:
        conf = yaml.safe_load(f)
    llm.LLMApi().init(conf)
    secret = conf.get("secret", None)
    if type(secret) is not str:
        raise ValueError("token must be configured and should be a string")
