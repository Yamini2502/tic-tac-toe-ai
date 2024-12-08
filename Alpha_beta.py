import time
from memory_profiler import memory_usage

def startPlayingGame():
    grid = [" " for _ in range(9)]
    playerTurn = True
    total_agent_moves = 0
    total_execution_time = 0
    total_memory_usage = 0

    print("\nWelcome to Tic Tac Toe! Playing against an AI agent using Alpha-Beta Pruning.")
    drawGrid(grid)

    while True:
        if playerTurn:
            move = getPlayersMove(grid)
            grid[move] = "X"
        else:
            print("\nAgent's turn:")
            start_time = time.time()
            mem_usage, result = memory_usage(
                (getAgentsMove, (grid,)), retval=True, interval=0.01
            )
            execution_time = time.time() - start_time
            agent_move = result
            total_execution_time += execution_time
            total_agent_moves += 1
            avg_memory = sum(mem_usage) / len(mem_usage)
            total_memory_usage += avg_memory

            grid[agent_move] = "O"
            print(f"Agent's move: {agent_move + 1}")
            print(f"Execution Time: {execution_time:.6f} seconds")
            print(f"Memory Usage for move: {avg_memory:.2f} MB")

        drawGrid(grid)

        if checkWinner(grid):
            if playerTurn:
                print("🎉 Congratulations! You won the game.")
            else:
                print("💡 The agent won the game. Better luck next time!")
            break
        elif checkDraw(grid):
            print("🤝 It's a draw! No more moves left.")
            break

        playerTurn = not playerTurn

    print("\nGame Performance Statistics:")
    print(f"Total Moves by Agent: {total_agent_moves}")
    print(f"Total Execution Time: {total_execution_time:.6f} seconds")
    print(f"Average Memory Usage per Move: {total_memory_usage / total_agent_moves:.2f} MB")

def drawGrid(grid):
    print("\nCurrent Board:")
    print(grid[0], "|", grid[1], "|", grid[2])
    print("---------")
    print(grid[3], "|", grid[4], "|", grid[5])
    print("---------")
    print(grid[6], "|", grid[7], "|", grid[8])

def getPlayersMove(grid):
    while True:
        move = input("\nYour turn! Enter a move (1-9): ")
        if move.isdigit() and int(move) in range(1, 10) and grid[int(move) - 1] == " ":
            return int(move) - 1
        else:
            print("Invalid move. Please try again.")

def getAgentsMove(grid):
    _, move = alphabeta(grid, True, -float("inf"), float("inf"))
    return move

def alphabeta(grid, isMaximizing, alpha, beta):
    if checkWinner(grid):
        return (-1 if isMaximizing else 1, None)
    elif checkDraw(grid):
        return (0, None)

    valid_moves = [i for i in range(9) if grid[i] == " "]
    
    # Move ordering: attempt to evaluate the most promising moves first.
    valid_moves.sort(key=lambda x: evaluateMove(grid, x, isMaximizing), reverse=True)

    if isMaximizing:
        bestScore = -float("inf")
        bestMove = None
        for i in valid_moves:
            grid[i] = "O"
            score, _ = alphabeta(grid, False, alpha, beta)
            grid[i] = " "
            if score > bestScore:
                bestScore = score
                bestMove = i
            alpha = max(alpha, bestScore)
            if beta <= alpha:
                break
        return (bestScore, bestMove)
    else:
        bestScore = float("inf")
        bestMove = None
        for i in valid_moves:
            grid[i] = "X"
            score, _ = alphabeta(grid, True, alpha, beta)
            grid[i] = " "
            if score < bestScore:
                bestScore = score
                bestMove = i
            beta = min(beta, bestScore)
            if beta <= alpha:
                break
        return (bestScore, bestMove)

def evaluateMove(grid, move, isMaximizing):
    # Example evaluation function (you can refine this based on heuristics)
    grid[move] = "O" if isMaximizing else "X"
    score = 0
    if checkWinner(grid):
        score = 1 if isMaximizing else -1
    grid[move] = " "
    return score

def checkWinner(grid):
    winningMoves = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for x, y, z in winningMoves:
        if grid[x] == grid[y] == grid[z] != " ":
            return True
    return False

def checkDraw(grid):
    return " " not in grid

if __name__ == "__main__":
    startPlayingGame()
