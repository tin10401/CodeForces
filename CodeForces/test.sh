#!/usr/bin/env bash

# 1) Compile all
g++ -std=c++20 generator.cpp -o generator
g++ -std=c++20 brute.cpp -o brute
g++ -std=c++20 submit.cpp -o submit

# 2) Decide how many tests to run
for ((i=1;; i++)); do
    echo "Test $i"

    # 3) Generate input
    ./generator > input.txt

    # 4) Run brute
    ./brute < input.txt > brute_out.txt

    # 5) Run submit
    ./submit < input.txt > submit_out.txt

    # 6) Compare
    if ! diff -q brute_out.txt submit_out.txt; then
        echo "Mismatch found on test $i!"
        echo "==============="
        echo "Input was:"
        cat input.txt
        echo "--------------"
        echo "Expected output:"
        cat brute_out.txt
        echo "--------------"
        echo "Actual output:"
        cat submit_out.txt
        echo "==============="
        exit 0
    fi
done

echo "All tests passed!"

