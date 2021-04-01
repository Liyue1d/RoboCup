#!/bin/bash
DIR=`realpath .`
PAYLOAD_DIR="$DIR/../payload"
SHARED_DIRS="-v $PAYLOAD_DIR/penalty_RL/agent2DRL:/robocup/agent2DRL -v $DIR/RL:/robocup/RL -v $PAYLOAD_DIR/penalty_RL/rcssserver-15.3.0:/robocup/serverPenalty"
docker run -e DISPLAY -e QT_X11_NO_MITSHM=1 $SHARED_DIRS -v $HOME/.Xauthority:/home/developer/.Xauthority --net=host -it thenn42/penalty_rl /bin/bash
