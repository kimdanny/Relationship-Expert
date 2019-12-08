
from flask import Flask, request, make_response
from json import dumps
app = Flask(__name__)



@app.route("/get_advice",  methods=[ 'POST'])
def postAdvice():
    print(request.json)
    response = "Our AI has analysed your response and judges it would be \
best for you not to send that response"

    return response



if __name__ == "__main__":
    app.run()
