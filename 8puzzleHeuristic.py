def manhattan(state, goal):
    dist = 0
    for i in range(1, 9):  
        x1, y1 = divmod(state.index(i), 3)  
        x2, y2 = divmod(goal.index(i), 3) 
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist
def main():
    print("Enter the current state of the puzzle (0 for blank):")
    state = list(map(int, input().split()))
    if len(state) != 9:
        print("Please enter exactly 9 numbers (0–8) separated by spaces.")
        return
    goal = [1, 2, 3,
            4, 5, 6,
            7, 8, 0]
    heuristic = manhattan(state, goal)
    print("\nInput State:")
    for i in range(0, 9, 3):
        print(state[i], state[i + 1], state[i + 2])
    print("\nGoal State:")
    for i in range(0, 9, 3):
        print(goal[i], goal[i + 1], goal[i + 2])
    print(f"\nManhattan Distance Heuristic Value: {heuristic}")

if __name__ == "__main__":
    main()
