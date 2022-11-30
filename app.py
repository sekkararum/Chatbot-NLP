import sklearn
from process import preparation, generate_response, load_sbmptn, prediksi_sbmptn, load_snmptn, prediksi_snmptn
from flask import Flask, render_template, request

# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

#SBM - SNM
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/frame2')
def frame2():
    return render_template('frame2.html')

@app.route('/frame3')
def frame3():
    return render_template('frame3.html')

# load model dan scaler
load_sbmptn()

@app.route('/sbmptn')
def sbmptn():
    return render_template('SBMPTN_Prediction.html')

@app.route("/predict_sbmptn", methods=["POST"])
def predict_sbmptn():
    # menangkap data yang diinput user melalui form
    Skor_Utbk = int(request.form['Skor_Utbk'])
    Program_Studi = int(request.form['Program_Studi'])

    # melakukan prediksi menggunakan model yang telah dibuat
    data = [[Skor_Utbk, Program_Studi]]
    prediction_result = prediksi_sbmptn(data)
    return render_template('SBMPTN_Prediction.html', hasil_prediksi=prediction_result)

# load model dan scaler
load_snmptn()

@app.route('/snmptn')
def snmptn():
    return render_template('SNMPTN_Prediction.html')

@app.route("/predict_snmptn", methods=["POST"])
def predict_snmptn():
    # menangkap data yang diinput user melalui form
    Bahasa_Indonesia = int(request.form['Bahasa_Indonesia'])
    Bahasa_Inggris = int(request.form['Bahasa_Inggris'])
    Biologi = int(request.form['Biologi'])
    Fisika = int(request.form['Fisika'])
    Kimia = int(request.form['Kimia'])
    Matematika = int(request.form['Matematika'])
    Program_Studi = int(request.form['Program_Studi'])

    # melakukan prediksi menggunakan model yang telah dibuat
    data = [[Bahasa_Indonesia, Bahasa_Inggris, Biologi, Fisika, Kimia, Matematika, Program_Studi]]
    prediction_result = prediksi_snmptn(data)
    return render_template('SNMPTN_Prediction.html', hasil_prediksi=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
