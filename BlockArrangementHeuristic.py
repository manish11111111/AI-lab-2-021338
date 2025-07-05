def print_state(state):
    print("State:", state)
def misplaced_blocks(state, goal):
    return sum(1 for i in range(len(state)) if state[i] != goal[i])
def distance_between_blocks(state, goal):
    distance = 0
    for i in range(len(state)):
        goal_index = goal.index(state[i])
        distance += abs(i - goal_index)
    return distance
def heuristic(state, goal):
    return misplaced_blocks(state, goal) + distance_between_blocks(state, goal)
def main():
    n = int(input("Enter the number of blocks: "))
    print("Enter the initial state of the blocks (space or comma-separated):")
    initial_state_input = input().replace(',', ' ')  
    initial_state = initial_state_input.split()
    print("Enter the goal state of the blocks (space or comma-separated):")
    goal_state_input = input().replace(',', ' ') 
    goal_state = goal_state_input.split()

    if len(initial_state) != n or len(goal_state) != n:
        print(f"Error: The state should contain exactly {n} blocks.")
        return

    print("\nInitial State:")
    print_state(initial_state)
    print("Goal State:")
    print_state(goal_state)

    print(f"Heuristic (misplaced + distance): {heuristic(initial_state, goal_state)}")


if __name__ == "__main__":
    main()
