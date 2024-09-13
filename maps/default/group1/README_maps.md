Group 1

Team Members:
Yoni Livstone, yl4944
Bin Choi, bc3159
Eylam Tagor, et2842

Contents:

- 2 json files: 1 easy and 1 hard map
- 2 python files that generated their respective json files
- This README file.

What makes a map difficult?

By definition, a "slow route" is when the route taken is significantly more steps than what an oracle algorithm would take.

There are a few factors that will make this traversal particularly difficult.

- the calculated step towards the shortest path without any walls is blocked
- there is an incomplete picture and the player is unaware of a unique or specific route to the "treasure"
  - it is hard to know the interval of any given wall
- traps designed to trick common heuristics for maze solving
- prime numbers which are more difficult to predict

What makes a map not difficult:

- taking many steps to complete the route
- taking an indirect route
