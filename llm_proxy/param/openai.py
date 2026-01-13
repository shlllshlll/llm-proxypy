#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: openai.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Self
import uuid
import time
import random
import string
from enum import Enum
import hashlib
from .common import *


def gen_id():
    return f"chatcmpl-{uuid.uuid4()}"


def gen_created():
    return int(time.time())


def gen_fp():
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    fp_str = hashlib.sha1(random_string.encode()).hexdigest()[:10]
    return f"fp-{fp_str}"


class FinishReason(Enum):
    stop = "stop"
    length = "length"
    content_filter = "content_filter"
    tool_calls = "tool_calls"


class Role(Enum):
    assistant = "assistant"
    user = "user"
    system = "system"
    tool = "tool"
    function = "function"
    developer = "developer"


class ResponseRole(Enum):
    assistant = "assistant"


@dataclass
class Function:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: Function
    index: int = 0
    type: str = "function"


@dataclass
class ChatRequest(ParamMixin):
    @dataclass
    class TextContentPart:
        text: str
        type: str = "text"

    @dataclass
    class ImageContentPart:
        @dataclass
        class ImageUrl:
            url: str
            detail: Hidden[str] = HIDE

        image_url: ImageUrl
        type: str = "image_url"

    @dataclass
    class AudioContentPart:
        class Format(Enum):
            wav = "wav"
            mp3 = "mp3"

        @dataclass
        class InputAudio:
            data: str
            format: "ChatRequest.AudioContentPart.Format"

        input_audio: InputAudio
        type: str = "input_audio"

    @dataclass
    class RefusalContentPart:
        type: str
        refusal: str

    @dataclass
    class DeveloperMessage:
        content: str | list["ChatRequest.TextContentPart"]
        role: Role = Role.developer
        name: Hidden[str] = HIDE

    @dataclass
    class SystemMessage:
        content: str | list["ChatRequest.TextContentPart"]
        role: Role = Role.system
        name: Hidden[str] = HIDE

    @dataclass
    class UserMessage:
        content: str | list["ChatRequest.TextContentPart | ChatRequest.ImageContentPart | ChatRequest.AudioContentPart"]
        role: Role = Role.user
        name: Hidden[str] = HIDE

    @dataclass
    class AssistantMessage:
        @dataclass
        class Audio:
            id: str

        @dataclass
        class Function:
            name: str
            arguments: str

        @dataclass
        class ToolCall:
            id: str
            type: str
            function: "ChatRequest.AssistantMessage.Function"

        content: str | list["ChatRequest.TextContentPart | ChatRequest.RefusalContentPart"]
        refusal: OptionHidden[str] = HIDE
        role: Role = Role.assistant
        name: Hidden[str] = HIDE
        audio: OptionHidden[Audio] = HIDE
        tool_calls: Hidden[list[ToolCall]] = HIDE
        function: OptionHidden[Function] = HIDE

    @dataclass
    class ToolMessage:
        content: str | list["ChatRequest.TextContentPart"]
        tool_call_id: str
        role: Role = Role.tool

    @dataclass
    class FunctionMessage:
        name: str
        content: Optional[str] = None
        role: Role = Role.function

    @dataclass
    class Prediction:
        content: str | list["ChatRequest.TextContentPart"]
        type: str = "content"

    @dataclass
    class Audio:
        class Voice(Enum):
            ash = "ash"
            ballad = "ballad"
            coral = "coral"
            sage = "sage"
            verse = "verse"
            alloy = "alloy"
            echo = "echo"
            shimmer = "shimmer"

        class Format(Enum):
            wav = "wav"
            mp3 = "mp3"
            flac = "flac"
            opus = "opus"
            pcm16 = "pcm16"

        voice: Voice
        format: Format

    @dataclass
    class ResponseFormat:
        class Type(Enum):
            text = "text"
            json_object = "json_object"
            json_schema = "json_schema"

        @dataclass
        class Schema:
            name: str
            description: Hidden[str] = HIDE
            schema: Hidden[Any] = HIDE
            strict: Hidden[bool] = HIDE

        type: Type
        json_schema: Hidden[Schema] = HIDE

    @dataclass
    class StreamOptions:
        include_usage: Hidden[bool] = HIDE

    @dataclass
    class Tools:
        @dataclass
        class Function:
            name: str
            parameters: Hidden[Any] = HIDE
            strict: OptionHidden[bool] = HIDE

        function: Function
        type: str = "function"

    @dataclass
    class ToolChoice:
        @dataclass
        class Function:
            name: str

        function: Function
        type: str = "function"

    @dataclass
    class FunctionCall:
        name: str

    @dataclass
    class Function:
        name: str
        description: Hidden[str] = HIDE
        parameters: Hidden[Any] = HIDE

    model: str
    messages: list[
        DeveloperMessage | SystemMessage | UserMessage | AssistantMessage | ToolMessage | FunctionMessage
    ] = field(default_factory=list)
    store: OptionHidden[bool] = HIDE
    metadata: Hidden[Any] = HIDE
    frequency_penalty: OptionHidden[float] = HIDE
    logit_bias: Hidden[Dict[str, float]] = HIDE
    logprobs: OptionHidden[bool] = HIDE
    top_logprobs: OptionHidden[int] = HIDE
    max_tokens: OptionHidden[int] = HIDE
    max_completion_tokens: OptionHidden[int] = HIDE
    n: OptionHidden[int] = HIDE
    modalities: OptionHidden[list[str]] = HIDE
    prediction: Hidden[Prediction] = HIDE
    audio: OptionHidden[Audio] = HIDE
    presence_penalty: OptionHidden[float] = HIDE
    response_format: Hidden[ResponseFormat] = HIDE
    seed: OptionHidden[int] = HIDE
    service_tier: OptionHidden[str] = HIDE
    stop: OptionHidden[str | list[str]] = HIDE
    stream: OptionHidden[bool] = HIDE
    stream_options: OptionHidden[StreamOptions] = HIDE
    temperature: OptionHidden[float] = HIDE
    top_p: OptionHidden[float] = HIDE
    tools: Hidden[list[Tools]] = HIDE
    tool_choice: Hidden[str | ToolChoice] = HIDE
    parallel_tool_calls: Hidden[bool] = HIDE
    user: Hidden[str] = HIDE
    function_call: Hidden[str | FunctionCall] = HIDE
    functions: Hidden[list[Function]] = HIDE

    @classmethod
    def create(cls, model: str, content: str, *args, **kwargs) -> Self:
        obj = cls(model, *args, **kwargs)
        obj.messages = [ChatRequest.UserMessage(content)]
        return obj


@dataclass
class StreamChatResponse(ParamMixin):
    @dataclass
    class Delta:
        content: OptionHidden[str] = ""
        function_call: OptionHidden[Function] = HIDE
        role: OptionHidden[ResponseRole] = ResponseRole.assistant
        refusal: OptionHidden[str] = None
        tool_calls: Hidden[list[ToolCall]] = HIDE

    @dataclass
    class Choice:
        delta: Optional["StreamChatResponse.Delta"] = None
        logprobs: float = None
        finish_reason: Optional[FinishReason] = None
        index: int = 0

    @dataclass
    class Usage:
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    model: str
    id: str = field(default_factory=gen_id)
    choices: list[Choice] = field(default_factory=list)
    created: int = field(default_factory=gen_created)
    service_tier: OptionHidden[str] = HIDE
    system_fingerprint: str = field(default_factory=gen_fp)
    object: str = "chat.completion.chunk"
    usage: OptionHidden[Usage] = HIDE

    @classmethod
    def create(cls, model: str, content: str | None, *args, **kwargs) -> Self:
        obj = cls(model, *args, **kwargs)
        if content is None:
            obj.choices = [StreamChatResponse.Choice(finish_reason=FinishReason.stop)]
        else:
            obj.choices = [StreamChatResponse.Choice(delta=StreamChatResponse.Delta(content=content))]
        return obj

    def to_line(self) -> str:
        return f"data: {self.to_json_str()}"

    @staticmethod
    def end_line() -> str:
        return "data: [DONE]"

    @staticmethod
    def header() -> Dict[str, str]:
        return {"Content-Type": "text/event-stream"}


@dataclass
class ChatResponse(ParamMixin):
    @dataclass
    class Audio:
        id: str
        expires_at: int
        data: str
        transcript: str

    @dataclass
    class Message:
        content: str
        refusal: Optional[str] = None
        tool_calls: Hidden[list[ToolCall]] = HIDE
        role: str = "assistant"
        function_call: Hidden[Function] = HIDE
        audio: Hidden["ChatResponse.Audio"] = HIDE

    @dataclass
    class Choice:
        message: "ChatResponse.Message"
        index: int = 0
        logprobs = None

    @dataclass
    class CompletionUsage:
        accepted_prediction_tokens: int = 0
        reasoning_tokens: int = 0
        rejected_prediction_tokens: int = 0
        audio_tokens: Hidden[int] = HIDE

    @dataclass
    class PromptUsage:
        cached_tokens: int = 0
        audio_tokens: Hidden[int] = HIDE

    @dataclass
    class Usage:
        completion_tokens: int = 0
        prompt_tokens: int = 0
        total_tokens: int = 0
        completion_tokens_details: "ChatResponse.CompletionUsage" = field(
            default_factory=lambda: ChatResponse.CompletionUsage()
        )
        prompt_tokens_details: "ChatResponse.PromptUsage" = field(default_factory=lambda: ChatResponse.PromptUsage())

    model: str
    id: str = field(default_factory=gen_id)
    choices: list[Choice] = field(default_factory=list)
    created: int = field(default_factory=gen_created)
    service_tier: OptionHidden[str] = HIDE
    system_fingerprint: str = field(default_factory=gen_fp)
    object: str = "chat.completion"
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def create(cls, model: str, content: str, *args, **kwargs) -> Self:
        obj = cls(model, *args, **kwargs)
        obj.choices = [ChatResponse.Choice(message=ChatResponse.Message(content=content))]
        return obj

    to_line = ParamMixin.to_json_str

    @staticmethod
    def header() -> Dict[str, str]:
        return {"Content-Type": "application/json"}
