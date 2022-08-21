#!/bin/env bash

if [[ -f data/fsps/README.md ]]
then
    echo "fsps files already downloaded"
else
  echo "downloading fsps files"
  git clone https://github.com/cconroy20/fsps.git data/fsps
fi
