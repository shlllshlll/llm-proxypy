#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: __main__.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief: 
"""
import os
import sys
import argparse
from pathlib import Path
import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import yaml
sys.path.append(str(Path(__file__).absolute().parent.parent))
from llm_proxy import auth, data, llm

logger = logging.getLogger()

def start_app(args):
    # 日志配置
    # 日志等级
    log_level_str = os.getenv('LOG_LEVEL', 'DEBUG')
    log_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=log_level)
    # 日志格式化
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s:%(name)s:%(module)s::%(filename)s:%(lineno)d %(message)s'
    )
    # 日志输出到文件，每小时归档
    log_path = Path(__file__).parent / "log"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    log_file = log_path / "server.log"
    file_handler = TimedRotatingFileHandler(log_file, when="H", interval=1, backupCount=240)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    # 设置各个模块的日志handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logger.info('Server starting...')

    # 读取配置
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    llm.LLMApi().init(conf)
    secret = conf.get("secret", None)
    if type(secret) is not str:
        raise ValueError("token must be configured and should be a string")

    if args.debug_server:
        if args.framework == 'flask':
            from llm_proxy import flask_server
            app = flask_server.run_app(args.host, args.port)
        else:
            from llm_proxy import fastapi_server
            app = fastapi_server.run_app(args.host, args.port)
    else:
        if args.framework == 'flask':
            from llm_proxy import flask_server
            app = flask_server.app
        else:
            from llm_proxy import fastapi_server
            app = fastapi_server.app
    
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Proxy')
    cmd_parsers = parser.add_subparsers(title='command', dest='command')

    # server
    server_parser = cmd_parsers.add_parser('server', help='start server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='host')
    server_parser.add_argument('--conf', type=str, default='conf/conf.yml', help='llm proxy server conf path')
    server_parser.add_argument('--port', type=int, default=8006, help='port')
    server_parser.add_argument('--no_debug_server', action='store_false', dest='debug_server', help='disable debug server')
    server_parser.add_argument('--framework', type=str, default='flask', choices=['flask','fastapi'],help='server framework')

    # auth
    auth_parser = cmd_parsers.add_parser('auth', help='authenticate user')
    auth_parser.add_argument('--username', type=str, default=None, help='username')
    auth_method_parsers = auth_parser.add_subparsers(title='method', dest='method')

    auth_gen_secret_parser = auth_method_parsers.add_parser('gen_secret', help='generate secrets')
    auth_gen_secret_parser.add_argument('--len', type=int, default=32, help='token length')

    auth_gen_token_parser = auth_method_parsers.add_parser('gen_token', help='generate token')
    auth_gen_token_parser.add_argument('secret', type=str, help='secret')
    auth_gen_token_parser.add_argument('--len', type=int, default=32, help='token length')
    auth_parser.add_argument('--expire', type=int, default=365 * 24, help='token expire time in hours')
    auth_valid_parser = auth_method_parsers.add_parser('valid', help='validate token')
    auth_valid_parser.add_argument('token', type=str, help='token')
    auth_valid_parser.add_argument('secret', type=str, help='secret')

    args = parser.parse_args()

    if args.command == 'auth':
        if args.method == "gen_secret":
            secret = auth.gen_secret(args.len)
            print(f'Secret: {secret}')
        elif args.method == 'gen_token':
            token = auth.gen_token(args.secret, args.expire, args.username)
            print(f'Token: {token}')
        elif args.method == 'valid':
            result = auth.verify_token(args.token, args.secret)
            print(f'Valid Result: {"Pass" if result else "Fail"}')
    elif args.command == 'server':
        app = start_app(args)
    else:
        raise ValueError('Invalid command')
