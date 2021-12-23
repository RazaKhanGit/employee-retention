from os import name
from flask import Flask, render_template, request

import predicitons

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        predict_data = []
        for i in request.form:
            # print(type(request.form[i]))
            predict_data.append(float(request.form[i]))
        willLeave = predicitons.make_predictions(predict_data)
        return render_template('index.html', prediction = willLeave)
    return render_template('index.html')

# @app.route("/sub", methods=['POST'])
# def sub():
#     name = ['h']
#     print('Post requested')
#     if request.method == "POST":
#         print('Post requested')
#         name = request.form['username']

#     return render_template('sub.html', n=name)

if __name__ == "__main__":
    app.run()