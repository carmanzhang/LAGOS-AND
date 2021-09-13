FROM python:3.6

LABEL maintainer="zlahu@foxmail.com"
EXPOSE 38081

RUN pip install nltk==3.5 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install flask==1.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scikit-learn==0.22.1 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scipy==1.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy==1.18.5 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install pandas==0.25.3 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install flask-cors==3.0.9 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install requests==2.24.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /

COPY lagos-and-rf-model.pkl lagos-and-rf-model.pkl
COPY author_similarity_infer.py author_similarity_infer.py

ENTRYPOINT ["python", "author_similarity_infer.py"]
