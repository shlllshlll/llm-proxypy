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
from typing import Any, Type, Optional, Coroutine
from shutils import static_vars

@static_vars(subclassdict={})
def get_class(cls: Type[Any], name: str) -> Optional[object]:
    def find_subclasses(cls: Type[Any]):
        subclasses = set(cls.__subclasses__())
        for subclass in subclasses.copy():
            subclasses.update(find_subclasses(subclass))
        return subclasses

    if cls not in get_class.subclassdict:
        subclasses_set = find_subclasses(cls)
        subclasses_dict = {}
        for subclass in subclasses_set:
            subclasses_dict[subclass.__name__] = subclass
        get_class.subclassdict[cls] = subclasses_dict
    else:
        subclasses_dict = get_class.subclassdict[cls]

    if name in subclasses_dict:
        return subclasses_dict[name]
    else:
        return None

def get_caller_class():
    return inspect.stack()[1].frame.f_locals.get('self', None).__class__

@static_vars(loop=None)
def get_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if get_event_loop.loop:
            loop = get_event_loop.loop
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            get_event_loop.loop = loop
    return loop

def run_coro_in_loop(coro: Coroutine, loop: asyncio.AbstractEventLoop):
    if loop.is_running():
        loop.create_task(coro)
    else:
        loop.run_until_complete(coro)
