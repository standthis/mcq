#!/bin/bash

set -e
rm -f results.txt
for i in extra/burst/*.png; do echo $i; ./mcq.py $i; done
#for i in extra/burst/*.png; do echo $i; ./mcq.py $i; feh research.png; done
#for i in extra/burst/*.png; do echo $i; ./mcq.py $i; done
