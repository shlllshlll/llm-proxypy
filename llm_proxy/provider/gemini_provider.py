#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: gemini_provider.py
Date: 2025/01/08 20:18:48
Author: shihaolei(shihaolei@baidu.com)
Modified By: shihaolei(shihaolei@baidu.com)
Last Modified: 2025/01/08 20:18:48
Copyright: (c) 2025 Baidu.com, Inc. All Rights Reserved
Brief:
"""

import re
import time
from random import choice
from typing import Dict, Tuple
import uuid
import json
from .provider import Provider, LLMError
from ..request_data import g
from ..sender import ResponseProtocol, Response


class GeminiProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def build_chat_request(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        super().build_chat_request(request_body)

        if g.stream:
            url = f"{self.base_url}/{g.model}:streamGenerateContent?alt=sse"
        else:
            url = f"{self.base_url}/{g.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": choice(self.token_list),
        }

        # 请求转换
        converted_body = {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
            "generationConfig": {},
            "contents": [],
        }

        # 模型参数
        generation_config = converted_body["generationConfig"]
        if "temperature" in request_body:
            generation_config["temperature"] = request_body.get("temperature")
        if "top_p" in request_body:
            generation_config["topP"] = request_body.get("top_p")
        if "max_tokens" in request_body:
            generation_config["maxOutputTokens"] = request_body.get("max_tokens")
        if "stop" in request_body:
            stop = request_body.get("stop")
            generation_config["stopSequences"] = stop if type(stop) != str else [stop]

        # tools或functioncall列表
        if "tools" in request_body:
            converted_body["function_declarations"] = [
                func for func in request_body["tools"]
            ]
        elif "functions" in request_body:
            converted_body["function_declarations"] = request_body["functions"]

        # system message
        request_messages = request_body.get("messages", [])
        start_idx = 0
        if len(request_messages) > 0 and request_messages[0]["role"] == "system":
            start_idx = 1
            converted_body["system_instruction"] = {
                "parts": {"text": request_messages[0]["content"]}
            }
        # other message
        converted_contents = converted_body["contents"]
        function_call_dict = {}
        for message in request_messages[start_idx:]:
            converted_contents.append({"role": "model", "parts": []})
            converted_content = converted_contents[-1]
            converted_parts = converted_content["parts"]

            message_role = message["role"]
            if message_role in ["user", "assistant"] and message.get("content"):
                converted_content["role"] = (
                    "user" if message_role == "user" else "model"
                )
                if type(message["content"]) == list:
                    for content in message["content"]:
                        if content["type"] == "text":
                            converted_parts.append({"text": message["content"]})
                        elif content["type"] == "image_url":
                            match = re.match(
                                r"data:(image/\w+);base64,(.*)",
                                message["image_url"]["url"],
                            )
                            if not match:
                                raise LLMError(
                                    f"only base64 image format supported for gemini, image_url: {message['image_url']['url']}."
                                )
                            image_type = match.group(1)
                            image_data = match.group(2)
                            converted_parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": f"image/{image_type}",
                                        "data": image_data,
                                    }
                                }
                            )
                else:
                    converted_parts.append({"text": message["content"]})
            elif message_role == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    converted_parts.append({"function_call": tool_call["function"]})
                    function_call_dict[tool_call["id"]] = tool_call["function"]["name"]
            elif message_role == "assistant" and message.get("function_call"):
                converted_parts.append({"function_call": message["function_call"]})
            elif message_role == "tool":
                function_name = function_call_dict.get(message["id"])
                if not function_name:
                    raise LLMError(f"tool_call id: {message['id']} not found.")
                converted_parts.append(
                    {
                        "function_response": {
                            "name": function_name,
                            "response": message["content"],
                        }
                    }
                )
            elif message_role == "function":
                converted_parts.append(
                    {
                        "function_response": {
                            "name": message["name"],
                            "response": message["content"],
                        }
                    }
                )

        return url, headers, converted_body

    def parse_chat_response(self, response: ResponseProtocol) -> ResponseProtocol:
        resp_json = response.json()
        converted_response = {
            "created": int(time.time()),
            "id": f"chatcmpl-{uuid.uuid4()}",
            "model": "gemini",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": resp_json["usageMetadata"]["candidatesTokenCount"],
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens": resp_json["usageMetadata"]["promptTokenCount"],
                "total_tokens": resp_json["usageMetadata"]["totalTokenCount"],
            },
            "choices": [],
        }

        for idx, candidate in enumerate(resp_json["candidates"]):
            choice = {
                "index": idx,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"refusal": None, "role": "assistant"},
            }
            converted_message = choice["message"]
            converted_response["choices"].append(choice)

            if len(candidate["content"]["parts"]) > 0:
                part = candidate["content"]["parts"][0]
                if "function_call" in part:
                    function_call = part["function_call"]
                    converted_message["tool_calls"] = [
                        {
                            "id": f"call_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "arguments": json.dumps(
                                    function_call["args"], ensure_ascii=False
                                ),
                                "name": function_call["name"],
                            },
                        }
                    ]
                else:
                    converted_message["content"] = part["text"]
            else:
                converted_message["content"] = ""
        return Response(
            json.dumps(converted_response, ensure_ascii=False).encode("utf8"),
            response.status_code,
            response.headers,
        )

    def parse_stream_chat_response(self, response: str, g_dict: Dict) -> str:
        response = response.strip()
        if response == "":
            return ""
        if response.startswith("data: "):
            response = response[len("data: ") :]

        resp_json = json.loads(response)
        if "id" not in g_dict:
            g_dict["id"] = f"chatcmpl-{uuid.uuid4()}"

        converted_response = {
            "id": g_dict["id"],
            "object": "chat.completion.chunked",
            "created": int(time.time()),
            "model": "gemini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": resp_json["candidates"][0]["content"]["parts"][0][
                            "text"
                        ],
                        "refusal": None,
                    },
                }
            ],
        }

        return f"data: {json.dumps(converted_response, ensure_ascii=False)}"
