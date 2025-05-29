import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


    def __init__(self):
        self.reset()

    
    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        self.total_lines_cleared = 0
        self.played_steps = 0
        # Penalty factors (can be tuned)
        self.hole_penalty_factor = 0.75 # 提升洞穴懲罰
        self.bumpiness_penalty_factor = 0.3 # 提升凹凸不平懲罰
        self.height_penalty_factor = 0.1 # 提高總高度懲罰係數
        self.max_height_penalty_factor = 0.15 # 新增最高列高度懲罰係數
        self.trapped_hole_penalty_factor = 0.5 # 提升被困洞穴懲罰
        # Combo clear tracking
        self.consecutive_line_clears = 0
        self.combo_reward_factor = 0.5 # Combo 獎勵因子
        return self._get_board_props(self.board)


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score

    def get_raw_stats(self):
        '''Returns the raw statistics for submission'''
        return self.total_lines_cleared, self.played_steps
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        num_cleared = len(lines_to_clear)

        if num_cleared > 0:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
            
            self.total_lines_cleared += num_cleared
            self.consecutive_line_clears += 1 # 增加連續清行計數
        else:
            self.consecutive_line_clears = 0 # 如果沒有清行，重置連續清行計數
            
        return num_cleared, board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    
    def _calculate_trapped_holes(self, board):
        '''Calculates the number of empty cells that have blocks above, to their left, and to their right.'''
        trapped_holes = 0
        for r in range(1, Tetris.BOARD_HEIGHT): # Start from row 1, needs a block above
            for c in range(Tetris.BOARD_WIDTH):
                if board[r][c] == Tetris.MAP_EMPTY and \
                   board[r-1][c] == Tetris.MAP_BLOCK: # Block above
                    
                    has_left_wall = (c > 0 and board[r][c-1] == Tetris.MAP_BLOCK)
                    has_right_wall = (c < Tetris.BOARD_WIDTH - 1 and board[r][c+1] == Tetris.MAP_BLOCK)
                    
                    # Trapped if walled on both sides, or at an edge and walled on the one available side
                    if (has_left_wall and has_right_wall) or \
                       (c == 0 and Tetris.BOARD_WIDTH > 1 and has_right_wall) or \
                       (c == Tetris.BOARD_WIDTH - 1 and Tetris.BOARD_WIDTH > 1 and has_left_wall):
                        trapped_holes += 1
        return trapped_holes


    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states


    def get_state_size(self):
        '''Size of the state'''
        return 4


    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        board_before_play = [row[:] for row in self.board] # For calculating properties before this piece
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        
        lines_cleared_this_step, self.board = self._clear_lines(self.board) # _clear_lines now updates self.total_lines_cleared
        
        current_props = self._get_board_props(self.board) # lines, holes, total_bumpiness, sum_height
        holes_after_play = current_props[1]
        bumpiness_after_play = current_props[2]
        # sum_height_after_play is current_props[3] but we also need max_height for new penalty
        
        # Get all height properties for reward calculation
        sum_h, max_h, _ = self._height(self.board) # sum_height, max_height, min_height

        # Base reward for placing a piece and clearing lines
        reward = 1 + (lines_cleared_this_step ** 2) * Tetris.BOARD_WIDTH

        # Combo reward
        if self.consecutive_line_clears > 1:
            combo_bonus = self.combo_reward_factor * (self.consecutive_line_clears -1)
            reward += combo_bonus
        
        # Penalties
        reward -= self.hole_penalty_factor * holes_after_play
        reward -= self.bumpiness_penalty_factor * bumpiness_after_play
        reward -= self.height_penalty_factor * sum_h # Use sum_h from _height()
        reward -= self.max_height_penalty_factor * max_h # Add penalty for max_height
        
        num_trapped_holes = self._calculate_trapped_holes(self.board)
        reward -= self.trapped_hole_penalty_factor * num_trapped_holes

        self.score += reward # Update cumulative score with the detailed reward
        self.played_steps += 1 # Increment played steps

        # Start new round
        self._new_round() # This might set self.game_over
        
        if self.game_over:
            reward -= 2 # Additional penalty for game over

        return reward, self.game_over


    def render(self):
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        display_text = f"Lines: {self.total_lines_cleared} Steps: {self.played_steps}"
        # Adjusted position (10, 22) and font size (0.6) for potentially longer text
        cv2.putText(img, display_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
