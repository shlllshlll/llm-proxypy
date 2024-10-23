#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: auth.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""

from typing import Optional
import secrets
import datetime
import logging
import jwt

logger = logging.getLogger(__name__)

def gen_secret(secret_len: int) -> str:
    return secrets.token_hex(secret_len)

def gen_token(secrets: str, expire: int = 365 * 24, username: Optional[str] = None) -> str:
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expire)
    }
    if username is not None:
        payload['username'] = username
    return jwt.encode(payload, secrets, algorithm='HS256')

def verify_token(token: str, secrets: str, username: str = None) -> bool:
    try:
        payload = jwt.decode(token, secrets, algorithms=['HS256'])
        if username is not None and payload.get('username', None) != username:
            logger.error("JWT username mismatch")
            return False
        return True
    except jwt.ExpiredSignatureError:
        logger.error("JWT token expired")
        return False
    except jwt.InvalidTokenError:
        logger.error("JWT token invalid")
        return False
