sudo docker run -it --rm -p 8889:8889 --gpus=all --memory=12g --memory-swap=24g\
 --mount type=bind,destination=/workspace/practice,source=`pwd`\
 tf2_practice_gpu
