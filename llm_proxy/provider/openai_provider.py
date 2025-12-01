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

import asyncio
import threading
from random import choice
from typing import Dict, Tuple
from shutils.cache import TTLCache
from .provider import Provider


class OpenAIProvider(Provider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        # refresh every hour
        self.models_interval = self.provider_config.get("models_interval", 3600)
        self._models_cache = TTLCache(ttl=self.models_interval)
        if self.models_interval > 0:
            self._models_cache.set("model_avaliable", 1)
        self._models_lock = threading.Lock()
        self._models_alock = asyncio.Lock()
        self.models_from_api = self.provider_config.get("models_from_api", False)
        if self.models_from_api:
            model_url = self.__get_model_url()
            response = self._request_sender.get(model_url, headers=self.__get_headers())
            if response.status_code != 200:
                raise Exception(f"Failed to get models from {model_url}")
            models = response.json()
            self._models = self.__process_response_data(models)

    def __get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {choice(self.token_list)}",
        }

    def build_chat_request(self, request_body: Dict) -> Tuple[str, Dict, Dict]:
        super().build_chat_request(request_body)

        url = f"{self.base_url}/chat/completions"
        headers = self.__get_headers()

        return url, headers, request_body


    def __should_fetch_remote(self) -> bool:
        if not self.models_from_api:
            return False
        elif self.models_interval > 0:
            if self._models_cache.get("model_avaliable"):
                return False
        else:
            return False

        return True

    def __get_model_url(self) -> str:
        if isinstance(self.models_from_api, str):
            return self.models_from_api
        return self.base_url + "/models"

    def __process_response_data(self, models_data: dict) -> set[str]:
        """处理 API 返回的数据并更新状态"""
        raw_list = models_data if isinstance(models_data, list) else models_data.get("data", [])
        new_models = set(model["id"] for model in raw_list)

        # 更新内存中的数据
        self._models = new_models
        self._models_cache.set("model_avaliable", 1)

        return self._models

    def models(self):
        if not self.__should_fetch_remote():
            return super().models()

        with self._models_lock:
            # check cache again to avoid duplicate fetch
            if self._models_cache.get("model_avaliable"):
                return super().models()

            model_url = self.__get_model_url()
            response = self._request_sender.get(model_url, headers=self.__get_headers())
            if response.status_code != 200:
                raise Exception(f"Failed to get models from {model_url}")
            models = response.json()
            model_list = self.__process_response_data(models)
        return self._get_nominal_models(model_list)

    async def async_models(self):
        if not self.__should_fetch_remote():
            return await super().async_models()

        async with self._models_alock:
            # check cache again to avoid duplicate fetch
            if self._models_cache.get("model_avaliable"):
                return self._get_nominal_models(self._models)

            model_url = self.__get_model_url()
            response = await self._request_sender.async_get(model_url, headers=self.__get_headers())
            if response.status_code != 200:
                raise Exception(f"Failed to get models from {model_url}")
            models = response.json()
            model_list = self.__process_response_data(models)

        return self._get_nominal_models(model_list)
