#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: __main__.py
Author: shlll(shlll7347@gmail.com)
Modified By: shlll(shlll7347@gmail.com)
Brief:
"""
import argparse
import logging
import uvicorn
from llm_proxy import auth, fastapi_server
from llm_proxy.settings import ServerSettings
from llm_proxy.config import init_llm

logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Proxy')
    cmd_parsers = parser.add_subparsers(title='command', dest='command')

    # server
    server_parser = cmd_parsers.add_parser('server', help='start server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='host')
    server_parser.add_argument('--conf', type=str, default='conf/conf.yml', help='llm proxy server conf path')
    server_parser.add_argument('--port', type=int, default=8006, help='port')
    server_parser.add_argument('--no_debug_server', action='store_false', dest='debug_server', help='disable debug server')
    server_parser.add_argument('--log_level', type=str, default='DEBUG', help='log level')

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


    match args.command:
        case 'auth':
            match args.method:
                case "gen_secret":
                    secret = auth.gen_secret(args.len)
                    logger.info(f'Secret: {secret}')
                case 'gen_token':
                    token = auth.gen_token(args.secret, args.expire, args.username)
                    logger.info(f'Token: {token}')
                case 'valid':
                    result = auth.verify_token(args.token, args.secret)
                    logger.info(f'Valid Result: {"Pass" if result else "Fail"}')
        case 'server':
            settings = ServerSettings(LOG_LEVEL=args.log_level, CONF=args.conf)
            fastapi_server.app.state.settings = settings
            init_llm()
            uvicorn.run("llm_proxy.app:app", host=args.host, port=args.port, reload=args.debug_server)
        case _:
            raise ValueError(f"Unknown command: {args.command}")
