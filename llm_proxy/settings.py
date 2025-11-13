#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: settings.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_PROXY_",
    )

    CONF: str = Field("conf/conf.yml")
    LOG_LEVEL: str = Field("WARNING")
