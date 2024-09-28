#!/bin/bash

# This script is used to run the tournament
# It will run the code of individual players on different maps in maps/tournament folder
# and generate the results in results/tournament folder

# Check if the player number is between 1 and 9
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <player_number>"
    exit 1
fi

if [ "$1" -lt 1 ] || [ "$1" -gt 9 ]; then
    echo "Player number should be between 1 and 9"
    exit 1
fi

# Check if the tournament folder exists
if [ ! -d "maps/tournament" ]; then
    echo "maps/tournament folder does not exist"
    exit 1
fi

# Check if the results folder exists for this player
if [ ! -d "results/tournament" ]; then
    mkdir results/tournament
fi

# Create a map for parameters to use with different maps
declare -A map
map[1]="maps/tournament/g1.json"
map[2]="maps/tournament/g2.json"
map[3]="maps/tournament/g3.json"
map[4]="maps/tournament/g4.json"
map[5]="maps/tournament/g5.json"
map[6]="maps/tournament/g6.json"
map[7]="maps/tournament/g7.json"
map[8]="maps/tournament/g9.json"

declare -A maximum_frequency
maximum_frequency[1]=9
maximum_frequency[2]=9
maximum_frequency[3]=13
maximum_frequency[4]=4
maximum_frequency[5]=10
maximum_frequency[6]=1
maximum_frequency[7]=13
maximum_frequency[8]=50

declare -A maximum_frequency
maximum_frequency2[1]=18
maximum_frequency2[2]=18
maximum_frequency2[3]=26
maximum_frequency2[4]=8
maximum_frequency2[5]=20
maximum_frequency2[6]=2
maximum_frequency2[7]=26
maximum_frequency2[8]=100

declare -A radius
radius[1]=5
radius[2]=20
radius[3]=40
radius[4]=150

# Run the player code on each map
for i in {7..7}
do
    for r in {4..4}
    do
      echo "Running player $1 on map $i with radius $r"
      python3 main.py -p $1 -mz ${map[$i]} -r ${radius[$r]} -m ${maximum_frequency[$i]} -ng --disable_logging > results/tournament/p${1}/p${1}_g${i}_r${r}_l_normal.json
      echo "Running player $1 on map $i with radius $r double l"
      python3 main.py -p $1 -mz ${map[$i]} -r ${radius[$r]} -m ${maximum_frequency2[$i]} -ng --disable_logging > results/tournament/p${1}/p${1}_g${i}_r${r}_l_double.json
      echo "Running player $1 on map $i with radius $r infinite l"
      python3 main.py -p $1 -mz ${map[$i]} -r ${radius[$r]} -m 100000000 -ng --disable_logging > results/tournament/p${1}/p${1}_g${i}_r${r}_l_infinity.json
    done
done