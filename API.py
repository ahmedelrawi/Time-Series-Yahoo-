from flask import Flask,render_template
from Time_Price_Yahoo import Time_Price_Yahoo
app = Flask(__name__)
tpy = Time_Price_Yahoo()


@app.route("/",methods=['GET', 'POST'])
def home():
    return render_template('home.html')



@app.route("/plot")
def plot():
    return render_template('plot.html')



app.run(debug=True)