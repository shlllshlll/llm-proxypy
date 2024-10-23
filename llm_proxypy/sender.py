#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: sender.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""

import json
import logging
from typing import Dict, Any, TYPE_CHECKING, Protocol, Awaitable, Iterable, Iterator, AsyncIterator
from dataclasses import dataclass, field
import asyncio
from .utils import get_caller_class, get_event_loop, run_coro_in_loop
if TYPE_CHECKING:
    import requests
    import aiohttp
    import httpx

logger = logging.getLogger(__name__)


class ResponseProtocol(Protocol):
    text: None | str | Iterable[str]
    status_code: int
    headers: Dict
    def json(self) -> Dict:
        ...
    
    @property
    def ok(self) -> bool:
        ...

    @property
    def content(self) -> str:
        ...

class StreamResponseProtocol(ResponseProtocol):
    def iter_lines(self) -> Iterable[str]:
        ...

class AStreamResponseProtocol(ResponseProtocol):
    async def aiter_lines(self) -> Awaitable[Iterable[str]]:
        ...

@dataclass
class Response:
    text: None | str | Iterable[str]
    status_code: int = 200
    headers: Dict = field(default_factory=dict)

    def json(self):
        try:
            return json.loads(self.text)
        except Exception as e:
            logger.error(e)
            return {}
    
    @property
    def ok(self) -> bool:
        return not (400 <= self.status_code < 600)

    @property
    def content(self) -> str:
        if type(self.text) is bytes:
            return self.text.decode(
                encoding=self.headers.get("content-encoding", "utf-8"))
        elif type(self.text) is str:
            return self.text
        else:
            raise TypeError("Invalid text type")

class Sender(object):
    def post(self, url: str, headers: Dict, body: Dict, stream: bool) -> ResponseProtocol | StreamResponseProtocol:
        if get_caller_class() == Sender:
            raise NotImplementedError
        else:
            return asyncio.run(self.async_post())

    async def async_post(self, url: str, headers: Dict, body: Dict, stream: bool) -> Awaitable[ResponseProtocol | AStreamResponseProtocol]:
        if get_caller_class() == Sender:
            raise NotImplementedError
        else:
            return self.post(url, headers, body, stream)

class RequestsSender(Sender):
    class AsyncStreamContext(Response):
        def __init__(self, response: 'requests.Response'):
            super().__init__('', response.status_code, response.headers)
            self._response = response

        async def __aenter__(self) -> "RequestsSender":
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def aiter_lines(self):
            with self._response as r:
                for line in r.iter_lines():
                    yield line

    def __init__(self) -> None:
        import requests
        self._requests = requests

    def post(self, url: str, headers: Dict, body: Dict, stream: bool) -> ResponseProtocol | StreamResponseProtocol:
        return self._requests.post(url, headers=headers, json=body, stream=stream)

    async def async_post(self, url: str, headers: Dict, body: Dict, stream: bool) -> Awaitable[ResponseProtocol | AStreamResponseProtocol]:
        resp = await asyncio.to_thread(self._requests.post, url, headers=headers, json=body, stream=stream)
        if stream:
            return RequestsSender.AsyncStreamContext(resp)
        else:
            return resp
    
class AiohttpSender(Sender):        
    class SyncStreamContext(Response):
        def __init__(self, session: 'aiohttp.ClientSession' , url: str, headers: Dict, body: dict):
            import aiohttp
            super().__init__('', 200, {})
            self._aiohttp = aiohttp
            self._session = session
            self._url = url
            self._headers = headers
            self._body = body
        
        def __enter__(self) -> "AiohttpSender":
            self._loop = get_event_loop()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        async def _stream(self):
            async with self._session.post(self._url, headers=self._headers, json=self._body) as response:
                async for line in response.content:
                    yield line

        def iter_lines(self):
            async_gen = self._stream()

            while True:
                try:
                    yield self._loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break

    class AsyncStreamContext(Response):
        def __init__(self, session: 'aiohttp.ClientSession' , url: str, headers: Dict, body: dict):
            import aiohttp
            super().__init__('', 200, {})
            self._aiohttp = aiohttp
            self._session = session
            self._url = url
            self._headers = headers
            self._body = body

        async def __aenter__(self) -> "RequestsSender":
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def aiter_lines(self):
             async with self._session.post(self._url, headers=self._headers, json=self._body) as response:
                async for line in response.content:
                    yield line

    def __init__(self) -> None:
        import aiohttp
        self._aiohttp = aiohttp
        self._loop = get_event_loop()
        self._session = self._aiohttp.ClientSession(loop=self._loop)
    
    def __del__(self) -> None:
        if self._session is not None:
            run_coro_in_loop(self._session.__aexit__(None, None, None), self._loop)
    
    def post(self, url: str, headers: Dict, body: Dict, stream: bool) -> ResponseProtocol | StreamResponseProtocol:
        if stream:
            return AiohttpSender.SyncStreamContext(self._session, url, headers, body)
        else:
            return self._loop_context.loop.run_until_complete(self._async_post(url, headers, body))
                
    async def _async_post(self, url: str, headers: Dict, body: Dict) -> Awaitable[ResponseProtocol | AStreamResponseProtocol]:
        async with self._session.post(url, headers=headers, json=body) as response:
            text = await response.text()
            return Response(text, response.status, response.headers)
            
    async def async_post(self, url: str, headers: Dict, body: Dict, stream: bool) -> Awaitable[ResponseProtocol | AStreamResponseProtocol]:
        async with self._session.post(url, headers=headers, json=body) as response:
            if stream is False:
                text = await response.text()
                return Response(text, response.status, response.headers)
            else:
                return AiohttpSender.AsyncStreamContext(self._session, url, headers, body)

class HttpxSender(Sender):
    class SyncStreamContext(Response):
        def __init__(self, response: 'Iterator[httpx.Response]'):
            super().__init__('', 200, {})
            self._response = response
        
        def __enter__(self) -> "AiohttpSender":
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def iter_lines(self):
            with self._response as r:
                for line in r.iter_bytes():
                    yield line
    
    class AsyncStreamContext(Response):
        def __init__(self, response: 'Iterator[httpx.Response]'):
            super().__init__('', 200, {})
            self._response = response

        async def __aenter__(self) -> "RequestsSender":
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def aiter_lines(self):
            async with self._response as r:
                async for line in r.aiter_bytes():
                    yield line

    def __init__(self) -> None:
        import httpx
        self._httpx = httpx
        # monkey patch
        self._httpx.Response.ok = property(lambda self: not (400 <= self.status_code < 600))
        # sync client
        self._client = None
        # async client
        self._async_client = None
    
    def __del__(self) -> None:
        if self._client is not None:
            self._client.close()
        if self._async_client is not None:
            loop = get_event_loop()
            run_coro_in_loop(self._async_client.aclose())
    
    def post(self, url: str, headers: Dict, body: Dict, stream: bool) -> ResponseProtocol | StreamResponseProtocol:
        if self._client is None:
            self._client = self._httpx.Client()
        if stream:
            return HttpxSender.SyncStreamContext(self._client.stream("POST", url, headers=headers, json=body))
        else:
            return self._client.post(url, headers=headers, json=body)
    
    async def async_post(self, url: str, headers: Dict, body: Dict, stream: bool) -> Awaitable[ResponseProtocol | AStreamResponseProtocol]:
        if self._async_client is None:
            self._async_client = self._httpx.AsyncClient()
        if stream:
            return HttpxSender.AsyncStreamContext(self._async_client.stream("POST", url, headers=headers, json=body))
        else:
            return await self._async_client.post(url, headers=headers, json=body)