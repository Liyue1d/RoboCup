#!/bin/bash

docker run -e DISPLAY -e QT_X11_NO_MITSHM=1 -v $HOME/.Xauthority:/home/developer/.Xauthority --net=host -it thenn42/robocuprl_base /bin/bash
