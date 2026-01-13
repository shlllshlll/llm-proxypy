#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: volcengine.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""

from dataclasses import dataclass, field
from enum import Enum
from shutils.param import Hidden, HIDE
from . import openai

@dataclass
class ChatRequest(openai.ChatRequest):
    """chat request for doubao"""
    @dataclass
    class Thinking:
        class Type(Enum):
            DISABLED = "disabled"
            ENABLED = "enabled"
            AUTO = "auto"
        type: Type = Type.ENABLED

    class ReasoningEffort(Enum):
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    thinking: Hidden[Thinking] = HIDE
    reasoning_effort: Hidden[ReasoningEffort] = HIDE

@dataclass
class StreamChatResponse(openai.StreamChatResponse):
    @dataclass
    class Delta(openai.StreamChatResponse.Delta):
        reasoning_content: Hidden[str] = HIDE

    @dataclass
    class Choice(openai.StreamChatResponse.Choice):
        delta: "StreamChatResponse.Delta | dict" = field(default_factory=dict)  # type: ignore[reportIncompatibleVariableOverride]

    choices: list[Choice] = field(default_factory=list)  # type: ignore[reportIncompatibleVariableOverride]

@dataclass
class ChatResponse(openai.ChatResponse):
    @dataclass
    class Message(openai.ChatResponse.Message):
        reasoning_content: Hidden[str] = HIDE

    @dataclass
    class Choice(openai.ChatResponse.Choice):
        message: "ChatResponse.Message"  # type: ignore[reportIncompatibleVariableOverride]

    choices: list[Choice] = field(default_factory=list)  # type: ignore[reportIncompatibleVariableOverride]
