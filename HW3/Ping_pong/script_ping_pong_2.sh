#!/bin/bash

for i in `seq 2 6`
do
    mpirun -np $i ./ping_pong_2 >> text_$i.txt
done


