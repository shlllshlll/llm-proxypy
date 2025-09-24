#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: common.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""

from typing import Dict
from shutils.param import *

def gen_header(
    content_type_json: bool = False,
    content_type_event_stream: bool = False,
    accept_json: bool = False,
) -> Dict[str, str]:
    headers = {}
    if content_type_json:
        headers["Content-Type"] = "application/json"
    elif content_type_event_stream:
        headers["Content-Type"] = "text/event-stream"

    if accept_json:
        headers["Accept"] = "application/json"

    return headers
