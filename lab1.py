import heapq
from typing import List, Tuple, Optional, Dict, Set

class Vehicle:
    """
    Represents a vehicle (car or truck) on the parking lot grid.

    Attributes:
        name: Unique identifier (e.g., 'A', 'B', '1', 'R' for red car)
        row: Row position of anchor cell (1-indexed)
        col: Column position of anchor cell (1-indexed)
        orientation: 'H' for horizontal, 'V' for vertical
        size: 2 for cars, 3 for trucks

    Anchor cell conventions:
        - Horizontal vehicles: (row, col) is the LEFT-most cell
        - Vertical vehicles: (row, col) is the TOP-most cell
    """
    def __init__(self, name: str, row: int, col: int, orientation: str, size: int):
        self.name = name
        self.row = row
        self.col = col
        self.orientation = orientation
        self.size = size

    def occupies(self, r: int, c: int) -> bool:
        """
        Check if this vehicle occupies the given cell (r, c).

        Args:
            r: Row coordinate (1-indexed)
            c: Column coordinate (1-indexed)

        Returns:
            True if vehicle occupies cell (r, c), False otherwise
        """
        if self.orientation == 'H':
            # For horizontal: check if same row and column is within range [col, col+size)
            return self.row == r and self.col <= c < self.col + self.size
        else:
            # For vertical: check if same column and row is within range [row, row+size)
            return self.col == c and self.row <= r < self.row + self.size

    def get_occupied_cells(self) -> List[Tuple[int, int]]:
        """
        Get all cells occupied by this vehicle.

        Returns:
            List of (row, col) tuples representing occupied cells
        """
        cells = []
        if self.orientation == 'H':
            # Horizontal: same row, columns from col to col+size-1
            for c in range(self.col, self.col + self.size):
                cells.append((self.row, c))
        else:
            # Vertical: same column, rows from row to row+size-1
            for r in range(self.row, self.row + self.size):
                cells.append((r, self.col))
        return cells

    def moved(self, dr: int, dc: int) -> "Vehicle":
        """
        Create a new vehicle moved by the given offset.

        Args:
            dr: Row offset (positive = down, negative = up)
            dc: Column offset (positive = right, negative = left)

        Returns:
            New Vehicle instance at the shifted position
        """
        return Vehicle(self.name, self.row + dr, self.col + dc, self.orientation, self.size)

class State:
    """
    Represents a state of the Traffic Jam puzzle.

    Attributes:
        vehicles: List of all vehicles on the grid
        grid_size: Dimension of the square grid (typically 6)
        exit_row: Row coordinate of the exit door
        exit_col: Column coordinate of the exit door
        red_car: Reference to the red car vehicle
    """
    def __init__(self, vehicles: List[Vehicle], grid_size: int, exit_row: int, exit_col: int):
        self.vehicles = vehicles
        self.grid_size = grid_size
        self.exit_row = exit_row
        self.exit_col = exit_col
        self.red_car = None
        # Find and store reference to red car
        for v in vehicles:
            if v.name == 'R':
                self.red_car = v
                break

    def is_goal(self) -> bool:
        """
        Check if the goal state is reached.

        Goal: Red car occupies the exit cell.

        Returns:
            True if red car is at the exit, False otherwise
        """
        return self.red_car.occupies(self.exit_row, self.exit_col)
    
    def occupancy_map(self) -> Dict[Tuple[int, int], str]:
        """
        Build a mapping of each occupied grid cell to the vehicle name occupying it.

        Used to:
        - Quickly check if a move is legal (whether the next cell is empty)
        - Support heuristics that need to know which vehicles block the red car

        Returns:
            Dictionary mapping (row, col) tuples to vehicle names
        """
        occ: Dict[Tuple[int, int], str] = {}
        for v in self.vehicles:
            for cell in v.get_occupied_cells():
                # Mark this cell as occupied by vehicle v
                occ[cell] = v.name
        return occ

    def in_bound_vehicle(self, v: Vehicle) -> bool:
        """
        Check if all cells of a vehicle are within the grid boundaries.

        Args:
            v: Vehicle to check

        Returns:
            True if vehicle is fully inside the grid, False otherwise
        """
        for r, c in v.get_occupied_cells():
            # Check if cell (r,c) is within valid grid range [1, grid_size]
            if not (1 <= r <= self.grid_size and 1 <= c <= self.grid_size):
                return False
        return True

    def is_legal(self) -> bool:
        """
        Check if the current state is legal (all vehicles in bounds).

        Returns:
            True if state is valid, False otherwise
        """
        # Check that all vehicles are within grid boundaries
        for v in self.vehicles:
            if not self.in_bound_vehicle(v):
                return False
        return True

    def signature(self) -> Tuple[Tuple[str, int, int], ...]:
        """
        Generate a canonical signature for this state.

        Returns a sorted tuple of (name, row, col) for all vehicles.
        This ensures identical states have identical signatures regardless
        of vehicle ordering in the list.

        Returns:
            Sorted tuple of vehicle positions
        """
        # Create list of (name, row, col) tuples, sort them, then convert to tuple
        # This ensures two states with same vehicles in different order are equal
        return tuple(sorted((v.name, v.row, v.col) for v in self.vehicles))

    def __hash__(self):
        """
        Hash function for states (enables use in sets and dict keys).

        Returns:
            Integer hash based on vehicle positions
        """
        return hash(self.signature())

    def __eq__(self, other):
        """
        Check equality between two states.

        Args:
            other: Another state to compare

        Returns:
            True if states have identical vehicle positions, False otherwise
        """
        if not isinstance(other, State):
            return False
        return self.signature() == other.signature()

    def board(self) -> str:
        """
        Generate a text-based visualization of the current state.

        Returns:
            String representation of the grid with:
            - 'R' for red car cells
            - Unique characters for other vehicles
            - '.' for empty cells
            - '*' for the exit cell (if empty)
            - Border characters for grid boundaries
        """
        # Create empty grid filled with dots
        grid = []
        for row_num in range(self.grid_size):
            row = []
            for col_num in range(self.grid_size):
                row.append('.')
            grid.append(row)

        # Mark exit cell with * if it's empty
        if 1 <= self.exit_row <= self.grid_size and 1 <= self.exit_col <= self.grid_size:
            # Convert from 1-indexed (puzzle) to 0-indexed (array)
            grid[self.exit_row - 1][self.exit_col - 1] = '*'

        # Place vehicles on grid
        for v in self.vehicles:
            for r, c in v.get_occupied_cells():
                if 1 <= r <= self.grid_size and 1 <= c <= self.grid_size:
                    # Convert from 1-indexed (puzzle) to 0-indexed (array)
                    grid[r - 1][c - 1] = v.name

        # Build string representation
        lines = []

        # Add column numbers at top (  1 2 3 4 5 6)
        header = "  "
        for col_num in range(1, self.grid_size + 1):
            header = header + str(col_num) + " "
        lines.append(header)

        # Add top border ( +----------+)
        border = " +"
        for i in range(self.grid_size * 2 - 1):
            border = border + "-"
        border = border + "+"
        lines.append(border)

        # Add each row with row number (1|. . . A B B|)
        row_num = 1
        for row in grid:
            line = str(row_num) + "|"
            for cell in row:
                line = line + cell + " "
            line = line + "|"
            lines.append(line)
            row_num = row_num + 1

        # Add bottom border
        lines.append(border)

        # Combine all lines with newlines
        result = ""
        for i in range(len(lines)):
            result = result + lines[i]
            if i < len(lines) - 1:  # Don't add newline after last line
                result = result + "\n"

        return result
    
    def successors(self) -> List[Tuple[str, "State"]]:
        """
        Generate all legal successor states reachable by one move.

        Each move shifts one vehicle by exactly one grid unit in a valid direction
        (horizontal vehicles move left/right, vertical vehicles move up/down).

        Returns:
            List of (action_description, successor_state) tuples
        """
        # Build map of which vehicle occupies each cell
        occ = self.occupancy_map()
        succ: List[Tuple[str, State]] = []

        def cell_free(r: int, c: int, moving_name: str) -> bool:
            """
            Check if a cell is free for the given vehicle to move into.

            Args:
                r: Row coordinate
                c: Column coordinate
                moving_name: Name of vehicle attempting to move

            Returns:
                True if cell is in bounds and either empty or occupied by moving vehicle
            """
            # Check if cell is within grid boundaries
            if not (1 <= r <= self.grid_size and 1 <= c <= self.grid_size):
                return False
            # Cell is free if empty OR if it contains the vehicle that's trying to move
            return (r, c) not in occ or occ[(r, c)] == moving_name

        # Try moving each vehicle in each valid direction
        for idx, v in enumerate(self.vehicles):  # idx = index in list, v = vehicle
            if v.orientation == 'H':
                # Try moving left: check if cell immediately left of leftmost part is free
                left_cell = (v.row, v.col - 1)
                if cell_free(left_cell[0], left_cell[1], v.name):  # left_cell[0]=row, left_cell[1]=col
                    new_v = v.moved(0, -1)  # Move 0 rows, -1 column (left)
                    new_vehicles = self.vehicles.copy()  # Make a copy of vehicle list
                    new_vehicles[idx] = new_v  # Replace old position with new position
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} left", new_state))

                # Try moving right: check if cell immediately right of rightmost part is free
                right_cell = (v.row, v.col + v.size)  # col + size gives position after vehicle
                if cell_free(right_cell[0], right_cell[1], v.name):
                    new_v = v.moved(0, +1)  # Move 0 rows, +1 column (right)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} right", new_state))

            else:  # orientation is 'V'
                # Try moving up: check if cell immediately above topmost part is free
                up_cell = (v.row - 1, v.col)  # row - 1 gives position above vehicle
                if cell_free(up_cell[0], up_cell[1], v.name):
                    new_v = v.moved(-1, 0)  # Move -1 row (up), 0 columns
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} up", new_state))

                # Try moving down: check if cell immediately below bottommost part is free
                down_cell = (v.row + v.size, v.col)  # row + size gives position below vehicle
                if cell_free(down_cell[0], down_cell[1], v.name):
                    new_v = v.moved(+1, 0)  # Move +1 row (down), 0 columns
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} down", new_state))
        
        return succ



def UCS(state: State) -> int:
    """
    Uniform Cost Search heuristic (always returns 0).

    This makes A* behave like UCS, exploring states in order of path cost only.

    Args:
        state: Current state (unused)

    Returns:
        Always returns 0
    """
    return 0

def h1(state: State) -> int:
    """
    Heuristic 1: Distance of red car from exit.

    Calculates the Manhattan distance from the red car's current position
    to the exit row. Since the red car is vertical and can only move
    vertically, this is simply the difference in rows.

    Admissibility: This is admissible because:
    - The red car must move at least this many steps to reach the exit
    - It never overestimates since we ignore all obstacles

    Args:
        state: Current state

    Returns:
        Number of rows between red car and exit
    """
    if state.red_car is None:
        return 0

    # Calculate vertical distance to exit
    return abs(state.red_car.row - state.exit_row)

def h2(state: State) -> int:
    """
    Heuristic 2: Distance plus number of blocking vehicles.

    Calculates h1 plus the count of distinct vehicles blocking the red car's
    path to the exit. A vehicle blocks if it occupies any cell in the exit
    column between the red car and the exit.

    Admissibility: This is admissible because:
    - We need at least h1 moves for the red car to reach the exit
    - Each blocking vehicle must be moved at least once to clear the path
    - Total moves needed >= distance + blocking_count
    - We don't count vehicles that may block each other, so it may underestimate

    Args:
        state: Current state

    Returns:
        h1 + number of vehicles blocking the exit path
    """
    if state.red_car is None:
        return 0

    distance = abs(state.red_car.row - state.exit_row)
    blocking_vehicles = set()  # Use set to count unique vehicles only

    # Determine the range of rows between red car and exit
    min_row = min(state.red_car.row, state.exit_row)  # Get lower bound
    max_row = max(state.red_car.row, state.exit_row)  # Get upper bound

    # Find all vehicles blocking the exit column in the red car's path
    for r in range(min_row, max_row + 1):  # +1 because range is exclusive at end
        for vehicle in state.vehicles:
            if vehicle.name == 'R':
                continue  # Skip red car itself
            # Check if vehicle occupies a cell in the exit column
            if vehicle.occupies(r, state.exit_col):
                blocking_vehicles.add(vehicle.name)  # Add to set (duplicates ignored)

    return distance + len(blocking_vehicles)  # Count unique blocking vehicles


def A_star(initial_state: State, heuristic) -> Tuple[Optional[List[State]], int]:
    """
    A* search algorithm to find optimal solution path.

    Uses a priority queue ordered by f(n) = g(n) + h(n), where:
    - g(n) = path cost from initial state (number of moves)
    - h(n) = heuristic estimate to goal

    Args:
        initial_state: Starting state of the puzzle
        heuristic: Heuristic function that takes a State and returns int

    Returns:
        Tuple of (solution_path, states_explored) where:
        - solution_path: List of states from initial to goal (None if no solution)
        - states_explored: Number of states expanded during search
    """
    counter = 0  # Tie-breaker for heap (ensures FIFO for equal f-values)
    frontier = []  # Priority queue (min-heap) of states to explore
    # Heap entry: (f_value, counter, g_value, state, path)
    # Add initial state to frontier with f = h(initial) since g = 0
    heapq.heappush(frontier, (heuristic(initial_state), counter, 0, initial_state, [initial_state]))

    explored = set()  # Set of visited states (for fast lookup)
    states_explored = 0  # Count states expanded

    while frontier:
        # Get state with lowest f-value from heap
        # _ means we ignore f_value and action since we don't need them here
        _, _, g_value, current_state, path = heapq.heappop(frontier)

        # Check if we've reached the goal
        if current_state.is_goal():
            return path, states_explored

        # Skip if already explored (may happen due to duplicate entries in frontier)
        if current_state in explored:
            continue

        # Mark as explored and increment counter
        explored.add(current_state)
        states_explored += 1

        # Generate and process all successor states
        for _, successor in current_state.successors():  # _ ignores action description
            if successor not in explored:
                # Calculate costs: g(n) = path length, h(n) = heuristic, f(n) = g + h
                new_g = g_value + 1  # Each move has unit cost
                h = heuristic(successor)
                new_f = new_g + h

                # Add to frontier with new path
                counter += 1  # Increment tie-breaker
                new_path = path + [successor]  # Append successor to path
                heapq.heappush(frontier, (new_f, counter, new_g, successor, new_path))

    # No solution found (frontier exhausted)
    return None, states_explored

def create_state_1() -> State:
    """
    Create initial state for Puzzle 1.

    Configuration:
    - 5 cars (A, B, C, D, E)
    - 2 trucks (1, 2)
    - 1 red car (R) at position (4, 6)
    - Exit at top-right corner (1, 6)

    Returns:
        Initial state for puzzle 1
    """
    vehicles = [
        Vehicle('A', 1, 4, 'V', 2),  # Car 1: (1,4) to (2,4) - vertical
        Vehicle('B', 1, 5, 'H', 2),  # Car 2: (1,5) to (1,6) - horizontal
        Vehicle('C', 2, 5, 'H', 2),  # Car 3: (2,5) to (2,6) - horizontal
        Vehicle('D', 3, 5, 'H', 2),  # Car 4: (3,5) to (3,6) - horizontal
        Vehicle('1', 3, 2, 'H', 3),  # Truck 1: (3,2) to (3,4) - horizontal
        Vehicle('2', 5, 3, 'H', 3),  # Truck 2: (5,3) to (5,5) - horizontal
        Vehicle('E', 3, 1, 'V', 2),  # Car 5: (3,1) to (4,1) - vertical
        Vehicle('R', 4, 6, 'V', 2),  # Red Car: (4,6) to (5,6) - vertical
    ]
    return State(vehicles, 6, 1, 6)  # Exit at (1, 6)


def create_state_2() -> State:
    """
    Create initial state for Puzzle 2.

    Configuration:
    - 5 cars (A, B, C, D, E)
    - 4 trucks (1, 2, 3, 4)
    - 1 red car (R) at position (4, 5)
    - Exit at top (1, 5)

    Returns:
        Initial state for puzzle 2
    """
    vehicles = [
        Vehicle('A', 1, 3, 'V', 2),  # Car 1: (1,3) to (2,3) - vertical
        Vehicle('B', 3, 3, 'H', 2),  # Car 2: (3,3) to (3,4) - horizontal
        Vehicle('C', 3, 5, 'H', 2),  # Car 3: (3,5) to (3,6) - horizontal
        Vehicle('D', 4, 1, 'H', 2),  # Car 4: (4,1) to (4,2) - horizontal
        Vehicle('E', 6, 3, 'H', 2),  # Car 5: (6,3) to (6,4) - horizontal
        Vehicle('1', 1, 1, 'V', 3),  # Truck 1: (1,1) to (3,1) - vertical
        Vehicle('2', 1, 2, 'V', 3),  # Truck 2: (1,2) to (3,2) - vertical
        Vehicle('3', 1, 4, 'H', 3),  # Truck 3: (1,4) to (1,6) - horizontal
        Vehicle('4', 2, 4, 'H', 3),  # Truck 4: (2,4) to (2,6) - horizontal
        Vehicle('R', 4, 5, 'V', 2),  # Red Car: (4,5) to (5,5) - vertical
    ]
    return State(vehicles, 6, 1, 5)  # Exit at (1, 5)


def create_state_3() -> State:
    """
    Create initial state for Puzzle 3.

    Configuration:
    - 2 cars (A, B)
    - 6 trucks (1, 2, 3, 4, 5, 6)
    - 1 red car (R) at position (3, 6)
    - Exit at top-right corner (1, 6)

    Returns:
        Initial state for puzzle 3
    """
    vehicles = [
        Vehicle('A', 2, 4, 'V', 2),  # Car 1: (2,4) to (3,4) - vertical
        Vehicle('B', 2, 5, 'H', 2),  # Car 2: (2,5) to (2,6) - horizontal
        Vehicle('1', 1, 4, 'H', 3),  # Truck 1: (1,4) to (1,6) - horizontal
        Vehicle('2', 1, 3, 'V', 3),  # Truck 2: (1,3) to (3,3) - vertical
        Vehicle('3', 3, 2, 'V', 3),  # Truck 3: (3,2) to (5,2) - vertical
        Vehicle('4', 4, 1, 'V', 3),  # Truck 4: (4,1) to (6,1) - vertical
        Vehicle('5', 5, 4, 'H', 3),  # Truck 5: (5,4) to (5,6) - horizontal
        Vehicle('6', 4, 3, 'H', 3),  # Truck 6: (4,3) to (4,5) - horizontal
        Vehicle('R', 3, 6, 'V', 2),  # Red Car: (3,6) to (4,6) - vertical
    ]
    return State(vehicles, 6, 1, 6)  # Exit at (1, 6)


def print_solution_path(path: List[State], puzzle_name: str):
    """
    Print the complete solution path with state visualizations.

    Args:
        path: List of states from initial to goal
        puzzle_name: Name of the puzzle for display
    """
    print(f"path for {puzzle_name}")
    print(f"Total moves: {len(path) - 1}\n")

    for i, state in enumerate(path):
        print(f"Step {i}:")
        print(state.board())
        print()


def main():
    """
    Main function to solve all Traffic Jam puzzles and display results.

    Solves three puzzle configurations using three search strategies:
    1. Uniform Cost Search (UCS) - h(n) = 0
    2. A* with h1 - distance to exit
    3. A* with h2 - distance + blocking vehicles

    Outputs:
    - Performance comparison table
    - Complete solution path for one heuristic per puzzle
    """
    print("Lab 1: Traffic Jam Puzzle Solver")

    # Define puzzle configurations
    states = [
        ("Puzzle 1", create_state_1()),
        ("Puzzle 2", create_state_2()),
        ("Puzzle 3", create_state_3())
    ]

    # Define search strategies
    heuristics = [
        ("Uniform Cost Search (h=0)", UCS),
        ("A* with h1 (distance)", h1),
        ("A* with h2 (distance + blocking)", h2)
    ]

    # Store results for comparison
    all_results = []

    # Solve each puzzle with each heuristic
    for puzzle_name, initial_state in states:
        print(f"{puzzle_name}")
        print(f"Exit position: ({initial_state.exit_row}, {initial_state.exit_col})")
        print(f"\nInitial state:")
        print(initial_state.board())
        print()

        puzzle_results = []

        for heuristic_name, heuristic_func in heuristics:
            print(heuristic_name)
            solution_path, states_explored = A_star(initial_state, heuristic_func)

            if solution_path:
                # Path includes initial state, so subtract 1 to get number of moves
                num_moves = len(solution_path) - 1
                print(f"Solution found: {num_moves} moves, {states_explored} states explored")
                # Store results in dictionary for later comparison
                puzzle_results.append({
                    'heuristic': heuristic_name,
                    'moves': num_moves,
                    'explored': states_explored,
                    'path': solution_path
                })
            else:
                print(f"No solution found ({states_explored} states explored)")
                puzzle_results.append({
                    'heuristic': heuristic_name,
                    'moves': None,
                    'explored': states_explored,
                    'path': None
                })

        all_results.append((puzzle_name, puzzle_results))

    # Print performance comparison table
    print("performance comparison:")

    for puzzle_name, puzzle_results in all_results:
        print(f"{puzzle_name}:")
        print(f"  {'heuristic':<35} {'moves':<10} {'states explored'}")
        for result in puzzle_results:
            # If moves exists, convert to string; otherwise use "N/A"
            moves_str = str(result['moves']) if result['moves'] is not None else "N/A"
            # <35 means left-align in 35-character field so that we can make it look like a table
            print(f"  {result['heuristic']:<35} {moves_str:<10} {result['explored']}")
        print()

    # Print solution paths (all heuristics find the same optimal path)
    print("optimal solution paths:")

    for puzzle_name, puzzle_results in all_results:
        # All heuristics find the same optimal path, so just use the first one
        first_result = puzzle_results[0]

        # If a solution was found, print the path
        if first_result['path'] is not None:
            print_solution_path(first_result['path'], puzzle_name)

if __name__ == "__main__":
    main()