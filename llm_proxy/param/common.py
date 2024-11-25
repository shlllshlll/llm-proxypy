#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: common.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""
from inspect import isclass
import json
import importlib
from dataclasses import is_dataclass, fields
from typing import Any, ForwardRef, TypeVar, Union, Dict, get_origin, get_args, get_type_hints
from enum import Enum


class Hide:
    pass


HIDE = Hide()

T = TypeVar("T")
Hidden = Union[T, Hide]
OptionHidden = Union[T, Hide, None]

class ParamMixin:
    def to_json_str(self):
        return json.dumps(obj=asdict(self), ensure_ascii=False, default=json_serializer)

def deref_forwardref(forward_ref: ForwardRef, module_name: str) -> Any:
    def tmp() -> forward_ref:
        ...

    res =  get_type_hints(tmp, vars(importlib.import_module(module_name)))["return"]
    return res

def deref_typestr(type_str: str, module_name: str) -> Any:
    module = importlib.import_module(module_name)
    parts = type_str.split(".")
    obj = module
    for part in parts:
        obj = getattr(obj, part)
    return obj

def asdict(obj: Any) -> dict:
    def asdict_internal(obj: Any) -> Any:
        if not is_dataclass(obj):
            if isinstance(obj, (list, tuple, set)):
                result = []
                for item in obj:
                    result.append(asdict_internal(item))
                result = type(obj)(result)
            elif isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    result[k] = asdict_internal(v)
                result = type(obj)(result)
            elif type(obj) in set((str, bool, type(None), int, float)):
                result = obj
            elif isinstance(obj, Enum):
                result = obj.value
            else:
                raise ValueError(f"Unsupported type[{type(obj)}]")
        else:
            result = {}
            for field_item in fields(obj):
                key = field_item.name
                value = getattr(obj, key)
                if isinstance(value, Hide):
                    continue
                result[key] = asdict_internal(value)
        return result

    if not is_dataclass(obj):
        raise ValueError("Not a dataclass")

    return asdict_internal(obj)


def dict_to_dataclass(data: Dict[str, Any], cls: T) -> T:
    def process_value(field_type: Any, value: Any, strict: bool = False) -> Any:
        if isinstance(field_type, ForwardRef):
            field_type = deref_forwardref(field_type, cls.__module__)
        elif type(field_type) == str:
            field_type = deref_typestr(field_type, cls.__module__)

        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin == Union:
            for item_type in args:
                try:
                    return process_value(item_type, value, strict=True)
                except Exception:
                    continue
            for item_type in args:
                try:
                    return process_value(item_type, value, strict=False)
                except Exception:
                    continue
            raise ValueError(f"Convert value[{value}] to type[{field_type}] failed")
        elif origin == list:
            arg_type = args[0]
            result = []
            for item in value:
                result.append(process_value(arg_type, item))
            return result
        elif origin == dict:
            key_type = args[0]
            value_type = args[1]
            result = {}
            for k, v in value.items():
                result[process_value(key_type, k)] = process_value(value_type, v)
            return result
        elif origin == tuple:
            result = []

            if len(args) != len(value):
                raise ValueError(
                    f"Tuple length mismatch: expected {len(args)}, got {len(value)}"
                )

            for arg_type, arg_item in zip(args, value):
                result.append(process_value(arg_type, arg_item))
            return tuple(result)
        else:
            if is_dataclass(field_type):
                return dict_to_dataclass(value, field_type)
            elif issubclass(field_type, Enum):
                return type(value)
            elif field_type == Any:
                return value
            else:
                if type(value) == field_type or isinstance(value, field_type):
                    return value
                else:
                    if strict:
                        raise ValueError(
                            f"Value type mismatch: expected {field_type}, got {type(value)}"
                        )
                    return type(value)

    field_values = {}

    for field_item in fields(cls):
        if field_item.name in data:
            value = data[field_item.name]
            field_values[field_item.name] = process_value(field_item.type, value)

    return cls(**field_values)


def json_serializer(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def gen_header(
    content_type_json: bool = False,
    content_type_event_stream: bool = False,
    accept_json: bool = False,
) -> Dict[str, str]:
    headers = {}
    if content_type_json:
        headers["Content-Type"] = "application/json"
    elif content_type_event_stream:
        headers["Content-Type"] = "text/event-stream"

    if accept_json:
        headers["Accept"] = "application/json"

    return headers
