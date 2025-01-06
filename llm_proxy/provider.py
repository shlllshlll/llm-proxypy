#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: provider.py
Author: shlll(shlll@baidu.com)
Modified By: shlll(shlll@baidu.com)
Brief:
"""

import logging
from random import choice
import json
from typing import (
    Set,
    Dict,
    Generator,
    Tuple,
    AsyncGenerator,
    List,
    TYPE_CHECKING,
    Optional,
    Any,
)
import time
import re
import uuid
import threading
from .utils import get_caller_class
from .sender import ResponseProtocol, Sender, Response
from .request_data import g
from .param import openai as openai_param

if TYPE_CHECKING:
    from apscheduler.schedulers.base import BaseScheduler

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class Provider(object):
    def __init__(
        self, conf: Dict, sender: Sender, scheduler: Optional["BaseScheduler"]
    ):
        self._request_sender = sender
        self.conf = conf
        self._schedular = scheduler
        self.models = set()
        for model in conf.get("models", []):
            self.models.add(model)
        self.base_url = conf.get("base_url")
        self.token_list = []
        self.token_list.extend(conf.get("tokens", []))
        self.modify = conf.get("modify", {})

    def __chat_common(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        g.stream = g.ori_stream
        url, headers, body = self.build_chat_request(request_body)
        logger.debug(f"headers: {headers}, url: {url}, body: {body}")
        return url, headers, body

    def __stream_gen_common(self, line: bytes, g_dict: Dict) -> Tuple[List[str], bool]:
        logger.debug(f"stream response line: {line}")
        line = line.decode("utf-8")  # type: str
        lines = line.split("\n")
        parsed_lines = []
        has_done = False
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            parsed_line = self.parse_stream_chat_response(line, g_dict).strip()
            if (
                not has_done
                and parsed_line == openai_param.StreamChatResponse.end_line()
            ):
                has_done = True
            parsed_lines.append(parsed_line + "\n\n")

        return parsed_lines, has_done

    def __to_stream_gen_common(self, response: ResponseProtocol) -> str:
        response_json = response.json()
        response_json["object"] = "chat.completion.chunk"
        response_json["choices"][0]["logprobs"] = None
        response_json["choices"][0]["delta"] = {
            "content": response_json["choices"][0]["message"]["content"]
        }
        del response_json["choices"][0]["message"]
        line = json.dumps(response_json, ensure_ascii=False)
        return line

    def get_models(self) -> Set[str]:
        return self.models

    def chat(self, request_body: Dict) -> ResponseProtocol | Generator[str, None, None]:
        url, headers, body = self.__chat_common(request_body)
        return self.do_request(url, headers, body)

    async def async_chat(
        self, request_body: Dict
    ) -> ResponseProtocol | Generator[str, None, None]:
        url, headers, body = self.__chat_common(request_body)
        return await self.async_do_request(url, headers, body)

    def do_request(
        self, url: str, headers: Dict, body: Dict
    ) -> ResponseProtocol | Generator[str, None, None]:
        response = self._request_sender.post(
            url, headers=headers, body=body, stream=g.stream
        )
        if response.ok is False:
            logger.warn(
                f"response failed, response code: {response.status_code}, message: {response.text}"
            )
            return response
        if g.stream is True:

            def generate():
                has_done = False
                g_dict = {}
                with response as r:
                    for line in r.iter_lines():
                        lines, local_has_done = self.__stream_gen_common(line, g_dict)
                        has_done = local_has_done if not has_done else has_done
                        for line in lines:
                            yield line
                    if not has_done:
                        yield f"{openai_param.StreamChatResponse.end_line()}\n\n"

            return generate()
        else:
            response = self.parse_chat_response(response)
            if g.ori_stream is True and self.modify.get(g.model, {}).get(
                "response_config", {}
            ).get("stream", False):

                def generate():
                    line = self.__to_stream_gen_common(response)
                    yield f"data: {line}\n\ndata: [DONE]\n"

                return generate()
            else:
                return response

    async def async_do_request(
        self, url: str, headers: Dict, body: Dict
    ) -> ResponseProtocol | AsyncGenerator[str, None]:
        response = await self._request_sender.async_post(
            url, headers=headers, body=body, stream=g.stream
        )
        if response.ok is False:
            logger.warn(
                f"response failed, response code: {response.status_code}, message: {response.text}"
            )
            return response
        if g.stream is True:

            async def generate():
                has_done = False
                g_dict = {}
                async with response as r:
                    async for line in r.aiter_lines():
                        lines, local_has_done = self.__stream_gen_common(line, g_dict)
                        has_done = local_has_done if not has_done else has_done
                        for line in lines:
                            yield line
                    if not has_done:
                        yield f"{openai_param.StreamChatResponse.end_line()}\n\n"

            return generate()
        else:
            response = self.parse_chat_response(response)
            if g.ori_stream is True and self.modify.get(g.model, {}).get(
                "response_config", {}
            ).get("stream", False):

                async def generate():
                    line = self.__to_stream_gen_common(response)
                    yield f"data: {line}\n\ndata: [DONE]\n"

                return generate()
            else:
                return response

    def build_chat_request(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        if get_caller_class() == Provider:
            raise NotImplementedError
        else:
            if g.model in self.modify:
                modify_conf = self.modify[g.model]
                modify_request = modify_conf.get("request_override", {})
                request_config = modify_conf.get("request_config", {})
                for key, value in modify_request.items():
                    if (
                        type(value) == str
                        and value.startswith("eval(")
                        and value.endswith(")")
                    ):
                        value = eval(value[5:-1])
                    request_body[key] = value
                if request_config.get("remove_system_message", False) is True:
                    messages = request_body.get("messages", [])
                    if len(messages) > 0 and messages[0].get("role") == "system":
                        del messages[0]

                g.stream = request_body.get("stream", False)
            return "", {}, {}

    def parse_chat_response(self, response: ResponseProtocol) -> ResponseProtocol:
        return response

    def parse_stream_chat_response(self, response: str, g_dict: Dict) -> str:
        return response


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

        logger.error(f"shlll response header: {response.headers}")
        return Response(
            converted_response.to_json_str(),
            response.status_code,
            {"content-type": "application/json"},
        )


class FallbackProvider(OpenAIProvider):
    pass
