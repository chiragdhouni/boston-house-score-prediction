import json
from flask import flask,request,app,send_from_directory,render_template,send_file,make_response,jsonify,url_for
import pickle

filename="regmodel.pkl"
regressor=pickle.load(open(filename, 'rb'))
scalar=pickle.load(open("scalar.pkl", 'rb'))

app=flask.Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    data=request.json([data])
    print(data)
    print(data[0])
    normalize_data=scalar.transform(np.array(list(data)).reshape(-1,1))
    print(normalize_data)
    prediction=regressor.predict(normalize_data)
    print(prediction[0])
    return (jsonify(prediction))

if(__name__=="__main__"):
    app.run(debug='True')
    

 
     


