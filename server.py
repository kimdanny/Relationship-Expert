
from flask import Flask, request, make_response
from json import dumps
app = Flask(__name__)



@app.route("/get_advice",  methods=[ 'POST'])
def postAdvice():
    print("server code has bee called and request.json is about to shown")
    print(request.json)
    userText = request.json
    response = "Our AI has analysed your response and judges it would be \
best for you not to send that response. Your text was "+userText

    return response



if __name__ == "__main__":
    app.run()
