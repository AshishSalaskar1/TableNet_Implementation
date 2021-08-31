from flask import *
from model_methods import predict_and_extract
app = Flask(__name__)  
  
@app.route("/")
def home():
  return render_template("home.html",base_url=request.base_url)

@app.route("/upload",methods=['POST'])
def upload():
  f = request.files['file']
  print(f.filename)
  file_name = f.filename + f.filename.split(".")[-1]
  f.save(file_name)
  predict_and_extract(file_name)

  return send_from_directory("./", filename="output.zip", as_attachment=True)

    
app.run()
