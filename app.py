import json
from flask import flask,request,app,send_from_directory,render_template,send_file,make_response,jsonify,url_for
import pickle

filename="regmodel.pkl"
regressor=pickle.load(open(filename, 'rb'))
scalar=pickle.load(open("scalar.pkl", 'rb'))

app=flask.Flask(__name__)


@app.route('/predict_1',methods=['POST'])
def predict_1():
    data=[float(x) for x in request.form.values()]
    final_data=scalar.transform(np.array(data).reshape(-1,1))
    output=regressor.predict(final_data)
    print(output[0])
    
    return render_template('home.html',predicted_text="the output is {}".format(output))


if(__name__=="__main__"):
    app.run(debug='True')
    

 
     


