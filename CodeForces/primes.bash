#!/bin/bash
upper_bound=100
if [ $# -gt 0 ]; then
    upper_bound=$1
fi
for num in $(seq 2 $upper_bound); do
    factors=$(factor $num)
    factors_only=${factors#*:}
    trimmed_factors=$(echo $factors_only | tr -d ' ')
    if [[ "$trimmed_factors" == "$num" ]]; then
        echo $num
    fi
done

