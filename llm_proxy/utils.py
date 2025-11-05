#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: utils.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""
import inspect
import asyncio
from typing import Coroutine, Protocol, cast
from shutils import static_vars

class GetClassProtocol(Protocol):
    subclassdict: dict[type, dict[str, type]]

    def __call__[T](self, cls: type[T], name: str) -> type[T]:
        ...

@static_vars(subclassdict={})
def get_class[T](cls: type[T], name: str) -> type[T]:
    self = cast(GetClassProtocol, get_class)

    def find_subclasses(cls: type[T]):
        subclasses = set(cls.__subclasses__())
        for subclass in subclasses.copy():
            subclasses.update(find_subclasses(subclass))
        return subclasses

    if cls not in self.subclassdict:
        subclasses_set = find_subclasses(cls)
        subclasses_dict = {}
        for subclass in subclasses_set:
            subclasses_dict[subclass.__name__] = subclass
        self.subclassdict[cls] = subclasses_dict
    else:
        subclasses_dict = self.subclassdict[cls]

    if name not in subclasses_dict:
        raise ValueError(f"Class {name} not found in subclasses of {cls.__name__}")
    return subclasses_dict[name]


class GetCallerClassProtocol(Protocol):
    loop: asyncio.AbstractEventLoop | None

    def __call__(self) -> type | None:
        ...

@static_vars(loop=None)
def get_event_loop():
    self = cast(GetCallerClassProtocol, get_event_loop)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if self.loop:
            loop = self.loop
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
    return loop

def run_coro_in_loop(coro: Coroutine, loop: asyncio.AbstractEventLoop):
    if loop.is_running():
        loop.create_task(coro)
    else:
        loop.run_until_complete(coro)
