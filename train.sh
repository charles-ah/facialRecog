#!/bin/sh
 
#echo "hello world"

x=0

while ((x<20))
do
    python recognition.py haarcascade_frontalface_default.xml
    echo $x
    ((x=x+1))
#((x++))
done