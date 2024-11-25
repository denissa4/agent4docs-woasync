from flask_api import FlaskAPI, status
from flask import request, jsonify

from .nlsql.handler import parsing_text
from .nlsql.nlsql_typing import NLSQLAnswer
from .talk_to_docs.talk_to_docs_handler import *

import asyncio
import logging
import os

app = FlaskAPI(__name__)


@app.route("/nlsql-analyzer", methods=['POST'])
def post_nlsql():
    if os.getenv('DEBUG', '') == '1':
        logging.info('Get request')
    loop = asyncio.get_event_loop()
    if request.is_json:
        if os.getenv('DEBUG', '') == '1':
            logging.info('This is json request')
        nlsql_answer: NLSQLAnswer = loop.run_until_complete(parsing_text(request.json.get('channel_id', ''),
                                                                         request.json.get('text', '')))

        return nlsql_answer, status.HTTP_200_OK

    return '', status.HTTP_400_BAD_REQUEST


### Talk to docs feature ###

db = ()
envs = {}

async def init():
    global db
    global envs
    db, envs = await initialize()
asyncio.run(init())

@app.route("/upload", methods=['POST'])
def upload_handler():
    if 'files[]' not in request.files:
        return jsonify({'status': 'Something went wrong'}), 400
    files = request.files.getlist('files[]')  # Handle multiple files
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for file_storage in files:
        tasks.append(loop.create_task(handle_upload(file_storage)))
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return jsonify({'status': 'Files processed successfully'}), 200


@app.route("/query", methods=['POST'])
def query_handler():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if request.is_json:
        nlsql_answer: NLSQLAnswer = loop.run_until_complete(handle_query(request.json.get('channel_id', ''),    
                                                                         request.json.get('text', ''), db, envs))
        loop.close()
        return nlsql_answer, status.HTTP_200_OK
    loop.close()
    return '', status.HTTP_400_BAD_REQUEST

