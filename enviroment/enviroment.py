class Sudoku:
    def __init__(self,puzzle_string: str,solution_string: str):
        self.puzzle_string = puzzle_string
        self.board = self.create_board(puzzle_string)
        self.solution = self.create_board(solution_string)


    def print_board(self):
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                print("---------------------")
            for j in range(len(self.board[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(self.board[i][j])
                else:
                    print(str(self.board[i][j]) + " ", end="")
        print("\n")

    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def set_number(self, number, pos):
        self.board[pos[0]][pos[1]] = number
        

    def valid(self, num, pos):
        # Check row
        for i in range(len(self.board[0])):
            if self.board[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.board[i][j] == num and (i,j) != pos:
                    return False
        return True
    
    def create_board(self, puzzle_string: str):
        board = []
        for i in range(9):
            board.append([])
            for j in range(9):
                board[i].append(int(puzzle_string[i*9+j]))
        return board
    
    def solve_backtraking(self):
        empty_cell = self.find_empty()

        if not empty_cell:
            return True
        
        for number in range(1, 10):
            if self.valid(number, empty_cell):
                self.set_number(number, empty_cell)
                if self.solve_backtraking():
                    return True
                self.set_number(0, empty_cell)

        return False


