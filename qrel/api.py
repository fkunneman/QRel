import json
from flask import Flask, request

from qrel.modules import relate

app = Flask(__name__)

model = Relate()

@app.route("/relate", methods=['GET'])

def search():
    '''
    :return: return a selection of 5 related questions to a given target question, if applicable add the target question to questions in the database
    '''
    related = {'code': 400}

    # query, method = '', 'ensemble'
    # if 'q' in request.args:
    #     query = request.args['q'].strip()
    # if 'method' in request.args:
    #     method = request.args['method'].strip()

    # if request.method == 'GET':
    #     questions = model(query=query.strip(), method=method)
    #     questions = { 'code':200, 'result': questions }

    # return json.dumps(questions)