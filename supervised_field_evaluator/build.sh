#!/bin/bash

DIR=`dirname $0`
if [ "$DIR" != "." ]; then
  echo "You must run this script in it's directory !"
  exit
fi

# Create a symbolic link for all necessary payloads for this container

# the tar -h option follows symlinks
tar -czh . ../payload/base/ ../payload/agent2D/ ../payload/thrift.tar.gz --exclude 'data*' | docker build -t thenn42/supervised_field_evaluator -
