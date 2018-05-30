
train:
	CUDA_VISIBLE_DEVICES=1 python ./src/train.py

docker:
	docker build -t smoke_att .

