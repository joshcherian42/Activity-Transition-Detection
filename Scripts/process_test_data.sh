#!/bin/bash

ala() {
    
    window_size=$1

    path=()
    while IFS='/' read -ra ADDR; do
        for i in "${ADDR[@]}"; do
            path+=($i)
        done
    done <<< "$2"

    filename=${2##*/}

    cd Scripts
    pypy -c 'import generate_test_data, sys; generate_test_data.process_file_bash(sys.argv[1], sys.argv[2]);' "$filename" $window_size
    cd ..
}

export -f ala
start_dir=${1:-`pwd`}

find "$start_dir" -type f -name '*.csv' | parallel --progress ala $2 {}
