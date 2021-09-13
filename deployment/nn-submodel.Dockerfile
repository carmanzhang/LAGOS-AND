FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

LABEL maintainer="zlahu@foxmail.com"
EXPOSE 38080

RUN pip install torchtext==0.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install flask==1.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install flask-cors==3.0.9 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /
COPY .vector_cache .vector_cache
COPY match-checkpoint.pkl match-checkpoint.pkl
COPY content_similarity_infer.py content_similarity_infer.py

ENTRYPOINT ["python", "content_similarity_infer.py"]
