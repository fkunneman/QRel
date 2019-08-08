
import json
from flask import Flask, request

from qrel.modules import relate

app = Flask(__name__)

model = relate.Relate()

@app.route("/relate", methods=['GET'])
def search():
    '''
    :return: return a selection of 5 related questions to a given target question, if applicable add the target question to questions in the database
    '''

    related = {'code': 400}
        
    if request.method == 'GET':
        if not request.json or not 'text' in request.json or not 'id' in request.json:
            abort(400)
        if 'n' in request.json.keys():
            related = model(request.json['text'],request.json['id'],request.json['n'])
        else:
            related = model(request.json['text'],request.json['id'])

    return json.dumps(related)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
