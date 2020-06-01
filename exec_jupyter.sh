sudo docker run -it --rm -p 8889:8889\
 --mount type=bind,destination=/workspace/practice,source=`pwd`\
 tf2_pr
