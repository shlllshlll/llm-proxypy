#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: openai_provider.py
Date: 2025/01/08 20:17:28
Author: shihaolei(shihaolei@baidu.com)
Modified By: shihaolei(shihaolei@baidu.com)
Last Modified: 2025/01/08 20:17:28
Copyright: (c) 2025 Baidu.com, Inc. All Rights Reserved
Brief:
"""

from random import choice
from typing import Dict, Tuple
from .provider import Provider


class OpenAIProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

    def build_chat_request(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        super().build_chat_request(request_body)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {choice(self.token_list)}",
        }

        return url, headers, request_body
