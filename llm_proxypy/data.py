#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: data.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""

from typing import Dict
import logging
from enum import Enum
from pathlib import Path
import yaml
from .llm import LLMApi

# 读取配置
conf_dir = (Path(__file__).parent.parent / "conf").absolute()
conf_file_path = conf_dir / "conf.yml"
with open(conf_file_path) as f:
    conf = yaml.safe_load(f)
LLMApi().init(conf)
secret = conf.get("secret", None)
if type(secret) is not str:
    raise ValueError("token must be configured and should be a string")

class ErrMsg(Enum):
    OK = 0
    ERROR = 1
    AUTH_ERROR = 2
    UNKNOWN_ERROR = 3
    RISK_CONTROL = 4

def resp(code: ErrMsg, msg: str, *args, **kwargs) -> Dict:
    response_data = {
        "code": code.value,
        "code_str": code.name,
        "msg": msg,
        "logid": "",
        "data": args or kwargs,
    }

    logging.debug(f"End of request, resp: {response_data}")
    return response_data

def get_bearer_token(auth_header: str) -> str:
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
    else:
        token = ""
    return token