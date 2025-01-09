#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: ernie_provider.py
Date: 2025/01/08 20:29:22
Author: shihaolei(shihaolei@baidu.com)
Modified By: shihaolei(shihaolei@baidu.com)
Last Modified: 2025/01/08 20:29:22
Copyright: (c) 2025 Baidu.com, Inc. All Rights Reserved
Brief:
"""

import json
from random import choice
import logging
from typing import Dict, List, Tuple
from .provider import Provider
from ..request_data import g
from ..param import openai as openai_param
from ..sender import ResponseProtocol, Response

logger = logging.getLogger(__name__)


class ErnieProvider(Provider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not self.base_url:
            self.base_url = (
                "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
            )
        self.get_token()

    def get_token(self):
        def request_token(ak_sk_list: List[Tuple[str, str]]):
            token_list = []
            for ak, sk in ak_sk_list:
                url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={ak}&client_secret={sk}"
                headers = openai_param.gen_header(
                    content_type_json=True, accept_json=True
                )
                resp = self._request_sender.post(url, headers, '""')
                token_list.append(resp.json().get("access_token"))
                logger.info(
                    f"Ernie Provider: get token Success for ak={ak}, sk={sk}, token={token_list[-1]}"
                )
            return token_list

        token_list = []
        ak_sk_list: List[Tuple[str, str]] = []
        for token_pair in self.token_list:
            if type(token_pair) == str:
                token_list.append(token_pair)
                continue
            ak, sk = token_pair
            ak_sk_list.append((ak, sk))
        if len(ak_sk_list) > 0:
            token_list += request_token(ak_sk_list)
            if self._schedular:
                self._schedular.add_job(
                    func=request_token, args=(ak_sk_list,), trigger="interval", days=29
                )
        self.token_list = token_list

    def build_chat_request(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        super().build_chat_request(request_body)

        url = f"{self.base_url}/{request_body["model"]}?access_token={choice(self.token_list)}"
        headers = openai_param.gen_header(content_type_json=True)
        # openai_request = openai_param.dict_to_dataclass()
        body = {}

        messages = request_body.get("messages", [])
        if len(messages) > 0 and messages[0].get("role") == "system":
            system_message = messages[0]["content"]
            body["system"] = system_message
            del messages[0]
        body["messages"] = messages
        if "temperature" in request_body:
            body["temperature"] = request_body["temperature"]
        if "top_p" in request_body:
            body["top_p"] = request_body["top_p"]
        if "presence_penalty" in request_body:
            body["penalty_score"] = request_body["presence_penalty"]
        if "stream" in request_body:
            body["stream"] = request_body["stream"]
        if "stop" in request_body:
            body["stop"] = request_body["stop"]
        if "max_tokens" in request_body:
            body["max_output_tokens"] = request_body["max_tokens"]
        if "max_completion_tokens" in request_body:
            body["max_output_tokens"] = request_body["max_completion_tokens"]

        return url, headers, body

    def parse_chat_response(self, response: ResponseProtocol) -> ResponseProtocol:
        resp_json = response.json()
        converted_response = openai_param.ChatResponse.create(
            g.model,
            resp_json["result"],
            id=resp_json["id"],
            created=resp_json["created"],
        )
        converted_response.usage.completion_tokens = resp_json["usage"][
            "completion_tokens"
        ]
        converted_response.usage.prompt_tokens = resp_json["usage"]["prompt_tokens"]
        converted_response.usage.total_tokens = resp_json["usage"]["total_tokens"]

        return Response(
            converted_response.to_json_str(),
            response.status_code,
            {"content-type": "application/json"},
        )

    def parse_stream_chat_response(self, response: str, g_dict: Dict) -> str:
        if response.startswith("data: "):
            response = response[6:]
        if response.startswith("[DONE]"):
            return "data: [DONE]"
        response_json = json.loads(response)
        resp_id = response_json["id"]
        model_name = g.model
        created = response_json["created"]
        finish = response_json["is_end"]
        content = response_json["result"] if not finish else None
        resp = openai_param.StreamChatResponse.create(
            model_name, content, id=resp_id, created=created
        )

        return resp.to_line()
