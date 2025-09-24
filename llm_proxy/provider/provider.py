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
)
from shutils import get_caller_class
from ..sender import ResponseProtocol, Sender
from ..request_data import g
from ..param import openai as openai_param

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
        self.provider_config = conf.get("config", {})
        self._schedular = scheduler
        self.models = set()
        for model in conf.get("models", []):
            self.models.add(model)
        self.base_url = conf.get("base_url")
        self.token_list = []
        self.token_list.extend(conf.get("tokens", []))
        self.modify = conf.get("modify", {})

    def post_init(self):
        model_prefix = self.provider_config.get("model_prefix", "")
        if len(model_prefix) > 0:
            self.nominal_models = set(model_prefix + model for model in self.models)
        else:
            self.nominal_models = self.models
    
    def get_real_model(self, model: str) -> str:
        model_prefix = self.provider_config.get("model_prefix", "")
        if len(model_prefix) > 0:
            return model[len(model_prefix):]
        return model

    def __chat_common(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        g.stream = g.ori_stream
        logger.debug(f"request body: {request_body}")
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
        return self.nominal_models

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
            logger.debug(f"response: {response.content}")
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
