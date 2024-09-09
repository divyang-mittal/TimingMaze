# Project 1: Timing Maze Game

## Citation and License
This project belongs to Department of Computer Science, Columbia University. It may be used for educational purposes under Creative Commons **with proper attribution and citation** for the author TAs **Divyang Mittal (First Author), Raavi Gupta and the Instructor, Prof. Kenneth Ross**.

## Summary

Course: COMS 4444 Programming and Problem Solving (Fall 2024)  
Problem Description: https://www.cs.columbia.edu/~kar/4444f24/node18.html 
Course Website: https://www.cs.columbia.edu/~kar/4444f24/
University: Columbia University  
Instructor: Prof. Kenneth Ross  
Project Language: Python

### TA Designer for this project

Divyang Mittal

### Teaching Assistants for Course
1. Divyang Mittal
2. Raavi Gupta

### All course projects

## Installation

Requires **python3.10** or higher

```bash
pip install -r requirements.txt
```

To install tkinter on macOS, run the following command:
```bash
brew install python-tk@3.X
```

## Usage

To view all options use python3 main.py -h
```bash
python3 main.py [-m/--max_door_frequency] [-r/--radius] [-s/--seed] [-mz/--maze] [-sc/--scale] [-T/--turns] 
      [-ng/--no_gui] [-p/--player]
```

## Debugging

The code generates a `log/debug.log` (detailed), `log/results.log` (minimal) and `log\<player_name>.log` 
(logs from player) on every execution, detailing all the turns and steps in the game.
