sudo docker run -it --rm -p 8889:8889 --memory=4g --memory-swap=8g\
 --mount type=bind,destination=/workspace/practice,source=`pwd`\
 tf2_practice_cpu
