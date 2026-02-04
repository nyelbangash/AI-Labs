read me


need Python 3.6 or higher
No external libraries required 

how to run:

1. Open a terminal/command prompt
2. Navigate to the directory containing lab1.py
3. Run the program:

- python lab1.py 
- python3 lab1.py
we recommend doing a 
- python3 lab1.py > lab1_output.txt
as the output will be very long due to the state visualizations.


The program will:
1. Solve all 3 puzzle configurations
2. Use 3 different search strategies for each puzzle:
   - Uniform Cost Search (h=0)
   - A* with h1 (distance heuristic)
   - A* with h2 (distance + blocking vehicles heuristic)
3. Display a comparison table showing states explored for each method
4. Display the optimal solution length for each puzzle
5. Show the optimal sequence of moves for each puzzle 


The program will display:
- Initial state for each puzzle
- Number of states explored for each search method
- Comparison table of all results
- Optimal solution lengths
- Full step-by-step solution visualization for one example

Grid vis: 
- 'R' represents the red car (goal is to move it to the exit)
- 'A', 'B', 'C', 'D', 'E' represent other cars
- '1', '2', '3', '4', '5', '6' represent trucks
- '.' represents empty cells
- Grid is 6x6 with 1-indexed coordinates
EX:
Initial state:
  1 2 3 4 5 6 
 +-----------+
1|. . . A B B |
2|. . . A C C |
3|E 1 1 1 D D |
4|E . . . . R |
5|. . 2 2 2 R |
6|. . . . . . |
 +-----------+

The code is organized into:
1. Vehicle class - represents individual vehicles
2. State class - represents puzzle configurations
3. Heuristic functions (h0, h1, h2) - guide the search
4. A* search algorithm - finds optimal solutions
5. Helper functions - create puzzles and display results
6. Main function - orchestrates everything

