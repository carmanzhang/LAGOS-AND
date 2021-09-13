#!/bin/bash

sudo docker build -t carmanzhang/lagos_and_model:nn-submodel-1.0-alpha -f ./nn-submodel.Dockerfile .
sudo docker build -t carmanzhang/lagos_and_model:1.0-alpha -f ./ml-submodel.Dockerfile .
