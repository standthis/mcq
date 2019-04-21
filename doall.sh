#!/bin/bash

#set -e
for i in extra/burst/*.png; do echo $i; ./mcq.py $i; feh research.png; done
