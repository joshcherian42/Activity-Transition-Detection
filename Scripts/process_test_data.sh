#!/bin/bash

process() {
    
    window_size=$1
    overlap_size=$2

    path=()
    while IFS='/' read -ra ADDR; do
        for i in "${ADDR[@]}"; do
            path+=($i)
        done
    done <<< "$2"

    filename=${3##*/}

    cd Scripts
    echo $3
    echo 'hey'
    pypy -c 'import generate_test_data, sys; generate_test_data.process_file_bash(sys.argv[1], sys.argv[2], sys.argv[3]);' "$filename" $window_size $overlap_size
    cd ..
}

export -f process
start_dir=${1:-`pwd`}

find "$start_dir" -type f -name '*.csv' | parallel --progress process $2 $3 {}
