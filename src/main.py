import tvm
from tvm.contrib import graph_runtime
import numpy as np
import fastapi
import base64
import cv2
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app=fastapi.FastAPI(title="insight face ",version="v0.0.1",description="人脸特征接口")


def model_interface():
    ctx = tvm.cpu()
    batch_size = 1
    image_shape = (3, 112, 112)
    data_shape = (batch_size,) + image_shape
    path_lib = "../model/deploy_lib.tar"
    loaded_json = open("../model/deploy_graph.json").read()
    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_params = bytearray(open("../model/deploy_param.params", "rb").read())    
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    
    def inference(input_nd_array):
        input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))
        module.run(data=input_data)
        out_deploy = module.get_output(0).asnumpy()
        return out_deploy.flatten()

    return inference

class request(BaseModel):
    imageBase64: str


interface=model_interface()
@app.post("/face_feature")
def face_detect(req:request):

    imgData = base64.b64decode(req.imageBase64)
    nparr = np.fromstring(imgData, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    blob=np.expand_dims(frame,0)
    feature=interface(blob).tolist()
    return JSONResponse(
        status_code=fastapi.status.HTTP_200_OK,
        content={
            "isSuc":True, 
            "code":0, 
            "msg":"success!", 
            "res":{"feature":feature}
        }
    )

