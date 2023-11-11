import os
import pathlib

import numpy as np
import onnxruntime
import torch
from flask import Flask, jsonify, request
from transformers import RobertaTokenizer

APP_DIRECTORY = pathlib.Path(__file__).parent.resolve()
MODEL_DIRECTORY = APP_DIRECTORY / ".." / "models"

ROBERTA_MODEL_PATH = MODEL_DIRECTORY / "roberta-sequence-classification-9.onnx"

app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
session = onnxruntime.InferenceSession(str(ROBERTA_MODEL_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    input_ids = torch.tensor(
        tokenizer.encode(request.json[0], add_special_tokens=True)
    ).unsqueeze(0)
    if input_ids.requires_grad:
        numpy_function = input_ids.detach().cpu().numpy()
    else:
        numpy_function = input_ids.cpu().numpy()
    inputs = {session.get_inputs()[0].name: numpy_function(input_ids)}
    output = session.run(None, inputs)
    result = np.argmax(output)
    return jsonify({"grumpy": bool(result)})


if __name__ == "__main__":
    host = os.environ.get("GRUMPY_HOST", "0.0.0.0")
    port = os.environ.get("GRUMPY_PORT", "8080")
    app.run(host=host, port=port)
