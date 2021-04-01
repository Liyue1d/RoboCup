#!/bin/bash

DIR=`realpath .`
docker run -e DISPLAY -e QT_X11_NO_MITSHM=1 -v $DIR/../payload/supervised_field_evaluator/opuSCOM2D:/robocup/opuSCOM2D -v $DIR/src:/robocup/src --net=host -it thenn42/supervised_field_evaluator /bin/bash
