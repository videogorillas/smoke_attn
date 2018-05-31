
train_fusion:
        OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=1 python ./src/train_fusion.py

train_temporal:
        OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=1 python ./src/train_temporal.py

train:
        CUDA_VISIBLE_DEVICES=1 python ./src/train.py

docker:
        docker build -t smoke_att .
