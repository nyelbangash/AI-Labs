import heapq
from typing import List, Tuple, Optional, Dict, Set

class Vehicle:
    """
    name: unique id (e.g., 'A', 'B', '1', 'R')
    row, col: anchor cell
      - H vehicles: (row, col) is the LEFT-most cell
      - V vehicles: (row, col) is the TOP-most cell
    orientation: 'H' or 'V' - 'H' for horizontal, 'V' for vertical
    size: 2 (car) or 3 (truck)
    """
    def __init__(self, name: str, row: int, col: int, length: int, orientation: str, size:int):
        self.name = name
        self.row = row
        self.col = col
        self.orientation = orientation
        self.size = size

    def occupies(self, r: int, c: int) -> bool:
        if self.orientation == 'H':
            return self.row == r and self.col <= c < self.col + self.size
        else:
            return self.col == c and self.row <= r < self.row + self.size
    
    def get_occupied_cells(self) -> List[Tuple[int, int]]:
        cells = []
        if self.orientation == 'H':
            for c in range(self.col, self.col + self.size):
                cells.append((self.row, c))
        else:
            for r in range(self.row, self.row + self.size):
                cells.append((r, self.col))
        return cells
    
    """Returns a vehicle shifted by (dr, dc)"""
    def moved(self, dr: int, dc: int) -> "Vehicle":
        return Vehicle(self.bame, self.row + dr, self.col + dc, self.orientation, self.size)

class State:
    def __init__(self, vehicles: List[Vehicle], grid_size: int, exit_row: int, exit_col: int):
        self.vehicles = vehicles
        self.grid_size = grid_size
        self.exit_row = exit_row
        self.exit_col = exit_col
        self.red_car = None
        for v in vehicles:
            if v.name == 'R':
                self.red_car = v
                break
        return self.red_car.row == self.exit_row and self.red_car.col == self.exit_col
    
    """Goal is reached when the red car occupies an 'exit' cell (edge of the map)"""
    def is_goal(self) -> bool:
        return self.red_car.occupies(self.exit_row, self.exit_col)
    
    """Builds a dictionary mapping each occupied grid cell (row, col) to the vehicle currently occupying it
    Used to:
    - detect overlaos (which are invalid states)
    - quickly check if a move is legal (see if the next cell is empty)
    - support heuriostics that need to know which vehicles block the red car
    """
    def occupancy_map(self) -> Dict[Tuple[int, int], str]:
        occ: Dict[Tuple[int, int], str] = {}
        for v in self.vehicles:
            for cell in v.get_occupied_cells():
                if cell in occ:
                    raise ValueError(f"Overlap at cell {cell} between {v.name} and {occ[cell]}")
                occ[cell] = v.name
        return occ

    """Check if a vehicle is fully inside the grid, used to make sure don't do an invalid move"""
    def in_bound_vehicle(self, v:Vehicle) -> bool:
        for r, c in v.get_occupied_cells():
            if not (0 <= r < self.grid_size and 0<= c < self.grid_size):
                return False
        return True

    """Check if inbounds and no overlaps"""
    def is_legal(self) -> bool:
        try:
            _ = self.occupancy_map()
        except ValueError:
            return False
        for v in self.vehicles:
            if not self.in_bound_vehicle(v):
                return False
        return True

    """Sort vehicles by name where ordering doesn't matter"""
    def signature(self) -> Tuple[Tuple[str, int, int], ...]:
        return tuple(sorted((v.name, v.row, v.col) for v in self.vehicles))
    
    """Get ChatGPT or something to print a board to check how it works"""
    def board(self) -> str:
        return "Temp"
    
    """Generate all legal successor states by moving one vehicle exactly one cell"""
    def successors(self) -> List[Tuple[str, "State"]]:
        occ = self.occupancy_map()
        succ: List[Tuple[str, State]] = []

        """Helper function to check if a cell is empty"""
        def cell_free(r: int, c:int, moving_name: str) -> bool:
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                return False
            return (r,c) not in occ or occ[(r,c)] == moving_name
        
        for idx, v in enumerate(self.vehicles):
            if v.orientation == 'H':
                # left: need cell immedietly left ir left most part free
                left_cell = (v.row, v.col-1)
                if cell_free(left_cell[0], left_cell[1], v.name):
                    new_v = v.moved(0, -1)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name}", new_state))
                
                # right: need cell immedietly right of the right-most part free
                right_cell = (v.row, v.col + v.size)
                if cell_free(right_cell[0], right_cell[1], v.name):
                    new_v = v.moved(0, +1)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} right", new_state))


            else:  # orientation is 'V'
                # up: cell immediately above top-most free
                up_cell = (v.row - 1, v.col)
                if cell_free(up_cell[0], up_cell[1], v.name):
                    new_v = v.moved(-1, 0)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} up", new_state))

                # down: cell immediately below bottom-most free
                down_cell = (v.row + v.size, v.col)
                if cell_free(down_cell[0], down_cell[1], v.name):
                    new_v = v.moved(+1, 0)
                    new_vehicles = self.vehicles.copy()
                    new_vehicles[idx] = new_v
                    new_state = State(new_vehicles, self.grid_size, self.exit_row, self.exit_col)
                    if new_state.is_legal():
                        succ.append((f"{v.name} down", new_state))
        
        return succ



def UCS(state: State) -> int:
    return 0

#how far is the red car from the exit
def h1(state: State) -> int:
    if state.red_car is None:
        return 0

    # Calculate vertical distance to exit
    return abs(state.red_car.row - state.exit_row)

#how far the red car is from the exit plus number of blocking vehicles
def h2(state: State) -> int:
    if state.red_car is None:
        return 0

    distance = abs(state.red_car.row - state.exit_row)
    num_blocking_vehicles = 0

    # Check for blocking vehicles in the path of the red car
    for r in range(state.red_car.row + 1, state.exit_row + 1):
        for vehicle in state.vehicles:
            if vehicle != state.red_car and vehicle.occupies(r, state.red_car.col):
                num_blocking_vehicles += 1
                break

    return distance + num_blocking_vehicles


def A_star(initial_state: State, heuristic) -> Tuple[Optional[List[State]], int]:
    counter = 0
    frontier = []
    heapq.heappush(frontier, (heuristic(initial_state), counter, 0, initial_state, [initial_state]))

    explored = set()
    
def create_state_1() -> State:
    vehicles = [
        Vehicle('A', 1, 4, 'V', 2),  # Car 1
        Vehicle('B', 1, 5, 'H', 2),  # Car 2
        Vehicle('C', 2, 5, 'H', 2),  # Car 3
        Vehicle('D', 3, 5, 'H', 2),  # Car 4
        Vehicle('1', 3, 2, 'H', 3),  # Truck 1
        Vehicle('2', 5, 3, 'H', 3),  # Truck 2
        Vehicle('E', 3, 1, 'V', 2),  # Car 5
        Vehicle('R', 4, 6, 'V', 2),  # Red Car
    ]
    return State(vehicles, 6, 1, 6)  # Exit at (1, 6)


def create_state_2() -> State:
    vehicles = [
        Vehicle('A', 1, 3, 'V', 2),  # Car 1
        Vehicle('B', 3, 3, 'H', 2),  # Car 2
        Vehicle('C', 3, 5, 'H', 2),  # Car 3
        Vehicle('D', 4, 1, 'H', 2),  # Car 4
        Vehicle('E', 6, 3, 'H', 2),  # Car 5
        Vehicle('1', 1, 1, 'V', 3),  # Truck 1
        Vehicle('2', 1, 2, 'V', 3),  # Truck 2
        Vehicle('3', 1, 4, 'H', 3),  # Truck 3
        Vehicle('4', 2, 4, 'H', 3),  # Truck 4
        Vehicle('R', 4, 5, 'V', 2),  # Red Car
    ]
    return State(vehicles, 6, 1, 5)  # Exit at (1, 5)


def create_state_3() -> State:
    vehicles = [
        Vehicle('A', 2, 4, 'V', 2),  # Car 1
        Vehicle('B', 2, 5, 'H', 2),  # Car 2
        Vehicle('1', 1, 4, 'H', 3),  # Truck 1
        Vehicle('2', 1, 3, 'V', 3),  # Truck 2
        Vehicle('3', 3, 2, 'V', 3),  # Truck 3
        Vehicle('4', 4, 1, 'V', 3),  # Truck 4
        Vehicle('5', 5, 4, 'H', 3),  # Truck 5
        Vehicle('6', 4, 3, 'H', 3),  # Truck 6
        Vehicle('R', 3, 6, 'V', 2),  # Red Car
    ]
    return State(vehicles, 6, 1, 6)  # Exit at (1, 6)
