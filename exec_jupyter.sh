sudo docker run -it --rm -p 8888:8888\
 --mount type=bind,destination=/workspace/practice,source=`pwd`\
 tf2_pr
