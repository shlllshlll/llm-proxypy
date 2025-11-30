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
    @staticmethod
    def convert_input(request_body: dict) -> dict:
        ...

    @staticmethod
    def convert_output(response: dict) -> dict:
        ...

    @staticmethod
    async def convert_output_stream(response: AsyncIterable[str], model_name: str) -> AsyncGenerator[str]:
        yield ""
