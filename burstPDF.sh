#!/bin/bash 

if ! [ -d output ]
then    
    mkdir output
else 
    rm -r output
    mkdir output
fi

pdf=$PWD/$1
path=$PWD/output
cp $pdf $path
cd $path
pdftoppm `basename $1` sheet -png
