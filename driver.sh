#!/bin/bash

for png in output/*.png; do
    echo $png started
    ./mcq.py $png
    echo $png done
done

