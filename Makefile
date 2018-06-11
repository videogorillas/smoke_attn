
gpu ?= 1

train_fusion_v4:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_fusion_v4_vgg.py

train_fusion_v3:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_fusion_v3_inception.py

train_fusion_v2:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_fusion_v2_mobilenet.py

train_fusion:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_fusion.py

train_temporal:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_temporal.py

train_spacial:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${gpu} python ./src/train_spacial.py

docker:
	docker build -t smoke_att .

