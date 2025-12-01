#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: servicer.py
Date: 2025/11/30 06:22:20
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Last Modified: 2025/11/30 06:22:20
Copyright: (c) 2025 shlll. All Rights Reserved
Brief:
"""

from typing import AsyncGenerator, AsyncIterable
from ..sender import Response


class Servicer:
    def __init__(self, conf: dict):
        self.servicer_conf = conf.get("servicer", {})

    def convert_input(self, request_body: dict) -> dict:
        ...

    def convert_output(self, response: dict, model_name: str) -> dict:
        ...

    async def convert_output_stream(self, response: AsyncIterable[str], model_name: str) -> AsyncGenerator[str]:
        yield ""
