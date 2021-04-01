#!/bin/bash

DIR=`dirname $0`
if [ "$DIR" != "." ]; then
  echo "You must run this script in it's directory !"
  exit
fi

# the tar -h option follows symlinks
tar -czh . ../payload/base/ ../payload/agent2D/ ../payload/thrift.tar.gz | docker build -t thenn42/penalty_rl -
