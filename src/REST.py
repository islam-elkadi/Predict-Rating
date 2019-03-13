from GridSearch import GridSearch
from Classifier import Train, Test
from CreateDataSets import DataSets

from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error": "Bad request"}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}), 404)

@app.route("/", methods = ["GET"])
def testApp():
    return jsonify({"success": "true"})

@app.route("/Preprocessing", methods = ["GET"])
def Preprocessing():
    preprocess.createSet(split_ratio = 0.8)
    return jsonify({"Create training & test Sets": "Success"})

@app.route("/TrainFastText", methods = ["POST"])
def TrainFastText():
    kwargs = {
        "name": request.json.get("name"),
        "model" :request.json.get("model"),
        "verbose": request.json.get("verbose"),
        "minCount": request.json.get("minCount"), 
        "minCountLabel": request.json.get("minCountLabel"), 
        "wordNgrams": request.json.get("wordNgrams"),
        "bucket": request.json.get("bucket"),
        "minn": request.json.get("minn"),
        "maxn": request.json.get("maxn"), 
        "t": request.json.get("t"), 
        "label": request.json.get("label"), 
        "lr": request.json.get("lr"), 
        "lrUpdateRate": request.json.get("lrUpdateRate"), 
        "dim": request.json.get("dim"), 
        "ws": request.json.get("ws"), 
        "epoch": request.json.get("epoch"), 
        "neg": request.json.get("neg"), 
        "loss": request.json.get("loss"), 
        "thread": request.json.get("thread"),
        "pretrainedVectors": request.json.get("pretrainedVectors"), 
        "saveOutput": request.json.get("saveOutput"), 
        "cutoff": request.json.get("cutoff"), 
        "retrain": request.json.get("retrain"), 
        "qnorm": request.json.get("qnorm"), 
        "qout": request.json.get("qout"), 
        "dsub": request.json.get("dsub")}
    train.trainClassifier(**kwargs)
    return jsonify({"Create/Process Training Data": "Success"})

@app.route("/Test_Classifier", methods = ["POST"])
def Test_Classifier():
    name = request.json.get("name")
    results = test.testClassifier(name)
    return jsonify({"Results": results})

@app.route("/Grid", methods = ["POST"])
def Grid():
    wordNgrams = request.json.get("wordNgrams")
    bucket = request.json.get("bucket")
    lr = request.json.get("lr")
    loss = request.json.get("loss")
    dim = request.json.get("dim")
    epoch = request.json.get("epoch")

    kwargs = {
        "wordNgrams":wordNgrams,
        "bucket":bucket,
        "lr":lr,
        "loss":loss,
        "dim":dim,
        "epoch":epoch,
    }
    
    grid.grid_search(kwargs)
    return jsonify({"Gird Search": "Complete"})

if __name__ == "__main__":
    test = Test()
    train = Train()
    grid = GridSearch()
    preprocess = DataSets()
    app.run(host = "0.0.0.0", port = 5000, debug = True)
