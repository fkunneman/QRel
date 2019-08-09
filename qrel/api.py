__author__='FlorianKunneman'

import json
from flask import Flask, request

from qrel.modules import relate

# initialize Flask
app = Flask(__name__)

# initialize relatedness module
model = relate.Relate()

@app.route("/related", methods=['GET'])
def rel():
    """
    :return: return a selection of 5 related questions to a given target question, if applicable add the target question to questions in the database
    """

    related = {'code': 400} # initialize output
        
    if request.method == 'GET': # only works with GET
        if not request.json or not 'text' in request.json or not 'id' in request.json: # check if input (question text and question id) is correct
            abort(400)
        if 'n' in request.json.keys(): # check if number of candidates is given as additional argument
            related = model(request.json['text'],request.json['id'],request.json['n']) # apply question relatedness procedure with given n
        else:
            related = model(request.json['text'],request.json['id']) # apply question relatedness procedure with default n

    return json.dumps(related)

@app.route("/similar", methods=['GET'])
def sim():
    """
    :return: return 5 most similar questions to a given target question
    """

    similar = {'code': 400} # initialize output
        
    if request.method == 'GET': # only works with GET
        if not request.json or not 'text' in request.json: # check if input (question text) is correct
            abort(400)
        if 'model' in request.json.keys(): # check if model is given as additional argument (options are 'bm25', 'softcosine', 'trlm', 'ensemble')
            similar = model.most_similar(request.json['text'],request.json['model']) # apply question similarity procedure with given model
        else:
            similar = model.most_similar(request.json['text']) # apply question simlarity procedure with default model (ensemble)
    print('Done:',similar)

    return json.dumps(similar)

@app.route("/update", methods=['GET'])
def update():
    """
    update dataset by applying question relatedness to questions that were found related to new questions
    """

    if request.method == 'GET':
        model.update_candidates()

    return 'updated succesfully and written to files'

# start-up api, by running python api.py
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
