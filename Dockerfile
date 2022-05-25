FROM autogluon/autogluon:0.3.1-rapids0.19-cuda10.2-jupyter-ubuntu18.04-py3.7
RUN pip install hydra-core omegaconf joblib onnxruntime pandas skl2onnx 
CMD python -V