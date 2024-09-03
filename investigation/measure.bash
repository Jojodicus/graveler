#! /bin/bash

RESULTFILE=./investigation/results.csv
ITERATIONS=100\\\'000\\\'000\\\'000

cd ..
make clean

echo "offload,runtime" > $RESULTFILE

function measure() {
    DEFINES="-DITERATIONS=$ITERATIONS -DOFFLOAD=$1" make
    echo -n $1, >> $RESULTFILE
    echo $(./graveler | tail -n 1 | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2};?)?)?[mGK]//g" | awk '{print $8}') >> $RESULTFILE
    make clean
}

LC_NUMERIC=en_US.UTF-8 # fix seq comma instead of period
for i in $(seq 0.97 0.002 1)
do
    measure $i
done
