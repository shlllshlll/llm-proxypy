#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: claude.py
Date: 2025/11/30 06:19:36
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Last Modified: 2025/11/30 06:19:36
Copyright: (c) 2025 shlll. All Rights Reserved
Brief:
"""

import uuid
import json
import logging
from typing import AsyncGenerator, AsyncIterable
from shutils.param import dict_to_dataclass
from ..param.openai import ChatResponse, StreamChatResponse
from ..sender import Response
from .servicer import Servicer

logger = logging.getLogger(__name__)


class ClaudeServicer(Servicer):
    @staticmethod
    def convert_input(request_body: dict) -> dict:
        messages = request_body.get("messages", [])
        system_prompt = request_body.get("system")

        # 1. message转换
        openai_messages = []

        # 1.1 处理 System Prompt (Claude 是顶层参数，OpenAI 是 role: system)
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        # 1.2 遍历消息进行转换
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 情况 A: 纯文本内容
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            # 情况 B: 复杂内容块 (文本 + 图片)
            new_content_list = []
            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type")

                    if block_type == "text":
                        new_content_list.append({"type": "text", "text": block.get("text")})

                    elif block_type == "image":
                        # --- 图片格式转换核心逻辑 ---
                        # Claude: source = {"type": "base64", "media_type": "...", "data": "..."}
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            media_type = source.get("media_type")
                            base64_data = source.get("data")

                            # 拼接成 Data URI 格式
                            data_uri = f"data:{media_type};base64,{base64_data}"

                            new_content_list.append({"type": "image_url", "image_url": {"url": data_uri}})

                    # 如果有其他不支持的类型(如 document/pdf)，暂时忽略或记录日志
                    elif block_type == "document":
                        logger.warning(
                            "Received 'document' type (PDF), which is not fully supported by OpenAI API yet. Ignoring."
                        )

            openai_messages.append({"role": role, "content": new_content_list})

        # 2. tools转换
        openai_tools = []
        claude_tools = request_body.get("tools", [])
        for tool in claude_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema", {}),  # 关键映射
                    },
                }
            )

        openai_request = {
            "model": request_body["model"],
            "messages": openai_messages,
            "temperature": request_body.get("temperature", 0.7),
            "max_tokens": request_body.get("max_tokens", 2048),
            "stop": request_body.get("stop_sequences", None),
            "stream": request_body.get("stream", False),
        }

        if openai_tools:
            openai_request["tools"] = openai_tools
            openai_request["tool_choice"] = "auto"

        return openai_request

    @staticmethod
    def convert_output(response: dict) -> dict:
        # Implement conversion logic from internal format to Claude format
        if 'error' in response:
            return {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": json.dumps(response["error"])
                }
            }
        openai_res = dict_to_dataclass(response, ChatResponse)

        msg = openai_res.choices[0].message
        content_blocks = []
        stop_reason = "end_turn"

        # 1. 文本内容
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        # 2. 工具调用
        if type(msg.tool_calls) == list and len(msg.tool_calls) > 0:
            stop_reason = "tool_use"
            for tool_call in msg.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments),
                    }
                )

        return {
            "id": openai_res.id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": openai_res.model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": openai_res.usage.prompt_tokens,
                "output_tokens": openai_res.usage.completion_tokens,
            },
        }

    @staticmethod
    async def convert_output_stream(response: AsyncIterable[str], model_name: str) -> AsyncGenerator[str]:
        # Implement async conversion logic from internal format to Claude format
        # 1. Message Start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f"msg_{uuid.uuid4()}", 'type': 'message', 'role': 'assistant', 'model': model_name, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        # 状态追踪
        current_block_index = 0
        in_tool_call = False

        async for chunks in response:
            chunks = [chunk.strip() for chunk in chunks.split("\n") if chunk.strip()]
            for chunk in chunks:
                if chunk.startswith("data:"):
                    chunk = chunk[len("data:") :].strip()
                if chunk == "[DONE]":
                    continue

                chunk_dict = json.loads(chunk)
                if "error" in chunk_dict:
                    yield f"event: error\ndata: {chunk}\n\n"
                    return
                stream_response = dict_to_dataclass(chunk_dict, StreamChatResponse)

                delta = stream_response.choices[0].delta

                # --- A. 处理普通文本 ---
                if delta.content:
                    # 如果之前在处理 Tool，现在来了 Text (通常不会发生，OpenAI 一般是一次性发完一种)，
                    # 但为了严谨，如果刚开始，发送 block start
                    if current_block_index == 0 and not in_tool_call:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta.content}}, ensure_ascii=False)}\n\n"

                # --- B. 处理工具调用 (Tool Calls) ---
                if type(delta.tool_calls) == list and len(delta.tool_calls) > 0:
                    for tool_call in delta.tool_calls:
                        # OpenAI 的流式 tool_calls 比较特殊：
                        # 第一帧包含 id 和 function name
                        # 后续帧包含 arguments (json 片段)

                        # 1. 新的 Tool Call 开始
                        if tool_call.id:
                            # 如果之前有文本块，先结束它 (简化逻辑：假设文本和工具不混排，或者工具总是在文本后)
                            # 实际 Claude 是支持 text 块后接 tool_use 块的，这里 index+1
                            if current_block_index == 0:
                                # 结束之前的 text 块 (如果有)
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                current_block_index = 1

                            in_tool_call = True

                            # 发送 Tool Use Block Start
                            block_start = {
                                "type": "content_block_start",
                                "index": current_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": {},  # input 为空，后续通过 delta 填充 json
                                },
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

                        # 2. 参数片段 (Arguments Delta)
                        if tool_call.function.arguments:
                            # Claude 使用 'input_json_delta'
                            delta_event = {
                                "type": "content_block_delta",
                                "index": current_block_index,
                                "delta": {"type": "input_json_delta", "partial_json": tool_call.function.arguments},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

        # 循环结束
        if in_tool_call:
            # 结束 tool use block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_block_index})}\n\n"
            # 停止原因为 tool_use
            stop_reason = "tool_use"
        else:
            # 结束 text block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            stop_reason = "end_turn"

        # Message Delta (Usage & Stop Reason)
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
