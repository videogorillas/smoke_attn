
cudadev:=1

train_fusion_v2:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${cudadev} python ./src/train_fusion_v2.py

train_fusion:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${cudadev} python ./src/train_fusion.py

train_temporal:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${cudadev} python ./src/train_temporal.py

train_spacial:
	OPENCV_OPENCL_RUNTIME="" CUDA_VISIBLE_DEVICES=${cudadev} python ./src/train_spacial.py

docker:
	docker build -t smoke_att .

