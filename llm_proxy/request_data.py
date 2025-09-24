#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: request_data.py
Date: 2024/11/28 01:31:33
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Last Modified: 2024/11/28 01:31:33
Copyright: (c) 2024 Baidu.com, Inc. All Rights Reserved
Brief:
"""

from contextvars import ContextVar

class State:
    def __init__(self):
        self._storage = {}

    def __getattr__(self, name):
        try:
            return self._storage[name]
        except KeyError:
            raise AttributeError(f"'State' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_storage":
            super().__setattr__(name, value)
        else:
            self._storage[name] = value

class G:
    _g: ContextVar[State] = ContextVar("g", default=State())

    def __getattr__(self, name):
        state = self._g.get()
        return getattr(state, name)

    def __setattr__(self, name, value):
        state = self._g.get()
        setattr(state, name, value)


    def clear(self):
        self._g.set(State())

g = G()