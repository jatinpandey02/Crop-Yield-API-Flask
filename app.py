from flask import Flask,request,jsonify
from pickle import load
import numpy as np
app = Flask(__name__)

model = load(open('model.pkl','rb'))
enc = load(open('label.pkl','rb'))

@app.route('/')
def hello():

    rain = float(request.args['rain'])
    temp = float(request.args['temp'])
    pest = float(request.args['pest'])
    item = request.args['item']

    input = [enc.transform([item])[0],rain,pest,temp]
    input = np.array([input])

    pred = model.predict(input)
    print(pred)
    return jsonify({'data':pred[0]})

if __name__ == '__main__':
    app.run()