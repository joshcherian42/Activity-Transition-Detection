#!/bin/bash

process() {
    
    window_size=$1
    overlap_size=$2
    dataset=$3

    cd Scripts
    echo $4
    python -c 'import generate_test_data, sys; generate_test_data.process_file_bash(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]);' "$4" $window_size $overlap_size $dataset
    cd ..
}

export -f process

echo "${@:4}" | parallel -d " " --progress process $1 $2 $3 {}
