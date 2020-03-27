# openCV-path-finding
This python code solves a project question using openCV, template matching and backtracking which is given in the README.md.

The question is as follows:-
You are the project head of an amazon development team working on a robot-based packaging solution for the assistance of humans. The task for the robot is to visit all special locations in the warehouse avoiding static obstacles and returning to the initial position(0,0). If the robot can't visit all special location return the maximum count of special locations it can visit also return the path the robot took to reach special locations. An image is given as input which contains a grid of dimension  (m_row*n_col). Further, the grid can only have three possible values in them  (-,+, empty). Static obstacles denoted by "-" and special locations by "+" and nothing denoted by blank space

The robot can go from 1 cell to others in any of the four primary directions (up, down, east, west)

m_row,n_col<100

The input will be in the form of a grid image with randomly distributed '+' and '-'.

The solution approach has a backtracking based algorithm.
