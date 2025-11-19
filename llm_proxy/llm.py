#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: llm.py
Author: shlll(shlll@baidu.com)
Modified By: shlll(shlll@baidu.com)
Brief: llm分发函数
"""

import logging
import inspect
import traceback
import json
import time
import asyncio
from random import choice
from typing import Dict, List
from shutils import singleton
from shutils.rwlock import RWLock, AsyncRWLock
from .utils import get_class
from .sender import Response, ResponseProtocol, Sender
from .provider import Provider
from .request_data import g
from .param import openai as openai_param

logger = logging.getLogger()


@singleton
class LLMApi(object):
    def __init__(self):
        self.initialized = False
        self._sender = None
        self._secret: str = ""
        self._provider_lock = RWLock()
        self._provider_alock = AsyncRWLock()

    @property
    def secret(self) -> str:
        if not self.initialized:
            raise Exception("LLMApi not initialized")
        return self._secret

    def __get_models(self, model_list: list[str]) -> str:
        response_body = {
            "object": "list",
            "data": [],
        }

        for model in sorted(model_list):
            response_body["data"].append(
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai",
                }
            )
        return json.dumps(response_body, ensure_ascii=False)

    def __get_provider(self, request_body: Dict) -> Provider:
        model = request_body.get("model", "")
        provider = None
        with self._provider_lock.read():
            if model in self.model_provider_dict:
                provider = choice(self.model_provider_dict[model])
        if provider is None:
            logger.warning(
                f"no provider found for model {model}"
            )
            raise Exception(f"no provider found for model {model}")
        g.model = provider.get_real_model(model)
        request_body["model"] = g.model
        return provider

    async def __get_provider_async(self, request_body: Dict) -> Provider:
        model = request_body.get("model", "")
        provider = None
        async with self._provider_alock.read():
            if model in self.model_provider_dict:
                provider = choice(self.model_provider_dict[model])
        if provider is None:
            logger.warning(
                f"no provider found for model {model}"
            )
            raise Exception(f"no provider found for model {model}")
        g.model = provider.get_real_model(model)
        request_body["model"] = g.model
        return provider

    def __handle_exception(self, exc: Exception):
        logger.error(f"unexpected exception: {traceback.format_exc()}")
        if g.ori_stream:

            def error_gen(e: Exception):
                resp = openai_param.StreamChatResponse.create(
                    g.model, f"```json\n{e}\n```"
                )
                yield f"{resp.to_line()}\n\n"
                yield f"{resp.end_line()}\n\n"

            return Response(
                error_gen(exc),
                status_code=500,
                headers={"Content-Type": "text/event-stream"},
            )
        else:
            resp = openai_param.ChatResponse.create(g.model, str(exc))
            return Response(
                resp.to_json_str(),
                status_code=500,
                headers={"Content-Type": "application/json"},
            )

    @staticmethod
    def check_init(method):
        async def async_wrapper(self, *args, **kwargs):
            if not self.initialized:
                raise Exception("LLMApi not initialized")
            return await method(self, *args, **kwargs)

        def sync_wrapper(self, *args, **kwargs):
            if not self.initialized:
                raise Exception("LLMApi not initialized")
            return method(self, *args, **kwargs)

        if asyncio.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper

    def init(self, conf: Dict, **kwargs) -> None:
        self.conf = conf
        self._secret = conf.get("secret", "")
        request_timeout = conf.get("request_timeout", 60)
        if self.conf.get("enable", False) == False:
            return

        sender_name = self.conf.get("sender", "RequestsSender")
        self._sender = get_class(Sender, sender_name)(request_timeout)

        self.model_provider_dict: dict[str, List[Provider]] = {}
        self.provider_list: list[Provider] = []
        for provider_conf in self.conf["provider"]:
            provider_name = provider_conf["type"]
            provider = get_class(Provider, provider_name)(
                provider_conf, self._sender, **kwargs
            )
            if provider is None:
                raise Exception(f"provider[{provider_conf['type']}] not found")
            provider.post_init()
            self.provider_list.append(provider)
            for model in provider.get_models():
                if model in self.model_provider_dict:
                    self.model_provider_dict[model].append(provider)
                else:
                    self.model_provider_dict[model] = [provider]
        self.initialized = True

    @check_init
    def chat(self, request_body: Dict) -> Response:
        g.ori_stream = request_body.get("stream", False)

        try:
            provider = self.__get_provider(request_body)
            response = provider.chat(request_body)
        except Exception as e:
            return self.__handle_exception(e)

        if isinstance(response, Response):
            return response
        elif isinstance(response, ResponseProtocol):
            return Response(
                response.content,
                status_code=response.status_code,
                headers={"Content-Type": response.headers["Content-Type"]},
            )
        else:
            return Response(response, headers={"Content-Type": "text/event-stream"})

    @check_init
    async def async_chat(self, request_body: Dict) -> Response:
        g.ori_stream = request_body.get("stream", False)

        try:
            provider = await self.__get_provider_async(request_body)
            response = await provider.async_chat(request_body)
        except Exception as e:
            return self.__handle_exception(e)

        if isinstance(response, Response):
            return response
        elif isinstance(response, ResponseProtocol):
            return Response(
                response.content,
                status_code=response.status_code,
                headers={"Content-Type": response.headers["Content-Type"]},
            )
        else:
            return Response(response, headers={"Content-Type": "text/event-stream"})

    @check_init
    def models(self) -> Response:
        model_list: set[str] = set()
        model_provider_dict: dict[str, list[Provider]] = {}
        for provider in self.provider_list:
            models = provider.get_models()
            model_list.update(models)
            for model in models:
                if model in model_provider_dict:
                    model_provider_dict[model].append(provider)
                else:
                    model_provider_dict[model] = [provider]
        with self._provider_lock.write():
            self.model_provider_dict = model_provider_dict

        models_list = set([ model for provider in self.provider_list for model in provider.get_models() ])
        return Response(
            self.__get_models(list(models_list)),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )

    @check_init
    async def async_models(self) -> Response:
        tasks = [ provider.async_models() for provider in self.provider_list ]
        results = await asyncio.gather(*tasks)

        models_list = set()
        model_provider_dict: dict[str, list[Provider]] = {}

        for models, provider in zip(results, self.provider_list):
            models_list.update(models)
            for model in models:
                if model in model_provider_dict:
                    model_provider_dict[model].append(provider)
                else:
                    model_provider_dict[model] = [provider]
        async with self._provider_alock.write():
            self.model_provider_dict = model_provider_dict

        return Response(
            self.__get_models(list(models_list)),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )
