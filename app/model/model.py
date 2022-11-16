import onnxruntime as rt
import numpy as np
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load ONNX model
model = rt.InferenceSession(f"{BASE_DIR}/anemia-{__version__}.onnx")
input_name = model.get_inputs()[0].name
label_name = model.get_outputs()[0].name

def model_predict(data):
    
    # Parse the input data into a np array 
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    
    #Predict the results
    output = model.run([label_name], {input_name: model_input.astype(np.float32)})[0][0]
    return output

def get_config_file():
    return open(f"{BASE_DIR}/config.json")
    