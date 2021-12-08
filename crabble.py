import copy
from collections import namedtuple
import itertools
import math
import random
from string import ascii_uppercase

RUN_TESTS = True
RECOMPUTE_LEAVES = False
RECOMPUTE_DEFENSE = False

RACK_SIZE = 5
# The layout of "premium" tiles on the board.
# 1 = double word, 2 = double letter, 3 = triple letter
PREMIUMS = [[1,0,0,0,2,0,0,0,1],
            [0,1,0,0,0,0,0,1,0],
            [0,0,3,0,0,0,3,0,0],
            [0,0,0,2,0,2,0,0,0],
            [2,0,0,0,1,0,0,0,2],
            [0,0,0,2,0,2,0,0,0],
            [0,0,3,0,0,0,3,0,0],
            [0,1,0,0,0,0,0,1,0],
            [1,0,0,0,2,0,0,0,1]]

BOARD_SIZE = len(PREMIUMS)

VALUES = {'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4,
          'G': 2, 'H': 4, 'I': 1, 'J': 8, 'K': 5, 'L': 1,
          'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
          'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
          'Y': 4, 'Z': 10}

# 40 tiles
FREQS = {'A': 4, 'B': 1, 'C': 1, 'D': 1, 'E': 4, 'F': 1,
         'G': 1, 'H': 1, 'I': 3, 'J': 1, 'K': 1, 'L': 1,
         'M': 1, 'N': 2, 'O': 3, 'P': 1, 'Q': 1, 'R': 2,
         'S': 2, 'T': 2, 'U': 1, 'V': 1, 'W': 1, 'X': 1,
         'Y': 1, 'Z': 1}

# Bonus for using entire rack.
BINGO_BONUS = 20

# Options for strategies to return.
PLAY = 'PLAY'
EXCHANGE = 'EXCHANGE'
PASS = 'PASS'

# Structure for a valid tile play, holding:
## the total score of the play
## a list of tuples with where each tile was played, and tile identities
## a list of all the words scored
ValidPlay = namedtuple(
    'ValidPlay', ['score', 'edits', 'scored_words'])

# Read in the words.
f = open("words.txt", "r")
WORDS = set([l.strip() for l in f.readlines() if len(l) <= BOARD_SIZE + 1])

VOWELS = 'AEIOU'
LEAVES = {}

if not RECOMPUTE_LEAVES:
    f = open("leave_values.txt", "r")
    values_total = 0
    for l in f.readlines():
        leave, v = l.strip().split(',')
        LEAVES[leave] = float(v)
else:
    leave_counts = {}
    leave_totals = {}
    for filenum, ct in ((1, 250000), (2, 150000), (3, 250000), (4, 100000),
                        (5, 250000), (8, 250000), (9, 150000), (10, 250000),
                        (11, 100000), (12, 250000), (13, 250000), (14, 150000),
                        (15, 250000), (16, 200000), (17, 100000), (18, 250000),
                        (19, 250000), (20, 150000), (21, 100000), (22, 250000),
                        (23, 250000), (24, 100000), (25, 100000), (26, 250000),
                        (27, 250000), (28, 100000), (29, 250000), (30, 250000)):
        f = open("leaves_{}_{}.txt".format(filenum, ct), "r")
        d = [l.strip().split(',') for l in f.readlines()]
        for leave, total, count in d:
            if leave not in leave_counts:
                leave_counts[leave] = 0
                leave_totals[leave] = 0  
            leave_counts[leave] += int(count)
            leave_totals[leave] += int(total)
    for k in leave_counts.keys():
        if leave_counts[k] >= 10: # make sure data is representative-ish
            LEAVES[k] = round(leave_totals[k] / leave_counts[k], 1)
    AVERAGE_LEAVE_VALUE = sum(LEAVES.values()) / len(LEAVES)

    s = sorted([(v, k) for k, v in LEAVES.items()])
    print("25 least valuable leaves:")
    for i in range(25):
        print("{}: {}".format(s[i][1], s[i][0]))
    print("25 most valuable leaves:")
    for i in range(-25, 0, 1):
        print("{}: {}".format(s[i][1], s[i][0]))
    def nearby_racks(r):
        nearby = []
        for index in range(len(r)):
            for letter in ascii_uppercase:
                new_r = ''.join(sorted(r[0:index] + letter + r[index+1:]))
                if new_r in LEAVES:
                    nearby.append(new_r)
        return nearby
    def possible_rack(rack):
        letters = set([l for l in rack])
        for l in letters:
            if rack.count(l) > FREQS[l]:
                return False
        return True
    # Fill in missing leave values.
    estimates = {}
    for n in range(1, RACK_SIZE):
        for r in itertools.product(ascii_uppercase, repeat=n):
            rack = ''.join(sorted(r))
            if not possible_rack(rack):
                continue
            if rack not in LEAVES and rack not in estimates:
                nb = nearby_racks(rack)
                if not nb:
                    estimates[rack] = round(AVERAGE_LEAVE_VALUE,1)
                else:
                    estimates[rack] = round(
                        sum([LEAVES[rr] for rr in nb]) / len(nb), 1)
    for r, v in estimates.items():
        LEAVES[r] = v
    print("Guessed {} of {} leave values.".format(len(estimates), len(LEAVES)))
    f = open("leave_values.txt", "w")
    for k in sorted(LEAVES.keys()):
        f.write("{},{}\n".format(k, LEAVES[k]))

DEFENSE_PER_CELL = {}
DEFENSE_PER_PLAY = {}
def empty_defense_stats():
    return [[[0 for _ in range(BOARD_SIZE)] for __ in range(BOARD_SIZE)]
            for __ in range(2)]

if not RECOMPUTE_DEFENSE:
    DEFENSE_PER_CELL = eval(
        open("defense_per_cell_values.txt", "r").readline().strip())
    f = open("defense_per_play_values.txt", "r")
    for l in f.readlines():
        p, v = l.strip().split('\t')
        DEFENSE_PER_PLAY[eval(p)] = float(v)
else:
    per_cell_counts = empty_defense_stats()
    per_cell_totals = empty_defense_stats()
    per_play_counts = {}
    per_play_totals = {}
    for filenum, ct in ((8, 250000), (9, 150000), (10, 250000), (11, 100000),
                        (12, 250000), (13, 250000), (14, 150000), (15, 250000),
                        (16, 200000), (17, 100000), (18, 250000), (19, 250000),
                        (20, 150000), (21, 100000), (22, 250000), (23, 250000),
                        (24, 100000), (25, 100000), (26, 250000), (27, 250000),
                        (28, 100000), (29, 250000), (30, 250000)):
        cf = open("defense_per_cell_{}_{}.txt".format(filenum, ct), "r")
        cd_totals = eval(cf.readline().strip())
        cd_counts = eval(cf.readline().strip())
        for v in range(2):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    per_cell_counts[v][r][c] += cd_counts[v][r][c]
                    per_cell_totals[v][r][c] += cd_totals[v][r][c]
        pf = open("defense_per_play_{}_{}.txt".format(filenum, ct), "r")
        for play, total, count in [l.strip().split('\t') for l in pf.readlines()]:
            play = eval(play)
            if play not in per_play_counts:
                per_play_counts[play] = 0
                per_play_totals[play] = 0
            per_play_counts[play] += int(count)
            per_play_totals[play] += int(total)
    DEFENSE_PER_CELL = empty_defense_stats()
    for v in range(2):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                DEFENSE_PER_CELL[v][r][c] = round(
                    per_cell_totals[v][r][c] / per_cell_counts[v][r][c], 2)
    f = open("defense_per_cell_values.txt", "w")
    f.write("{}\n".format(DEFENSE_PER_CELL))
    for k, v in per_play_counts.items():
        if v >= 10: # make sure data is representative-ish
            DEFENSE_PER_PLAY[k] = round(per_play_totals[k] / v, 2)
    def possible_plays():
        plays = set()
        for lane in range(BOARD_SIZE):
            for n in range(1, RACK_SIZE+1):
                for comb in itertools.combinations(range(BOARD_SIZE), n):
                    for v in itertools.product(range(2), repeat=n):
                        across_play = tuple([(v[i], lane, comb[i]) for i in range(n)])
                        plays.add(across_play)
                        down_play = tuple([(v[i], comb[i], lane) for i in range(n)])
                        plays.add(down_play)
        return plays
    # Estimate missing values.
    estimated = 0
    for p in possible_plays():
        if p not in DEFENSE_PER_PLAY:
            estimated += 1
            DEFENSE_PER_PLAY[p] = round(sum(
                [DEFENSE_PER_CELL[v][r][c] for v, r, c in p]) / len(p), 2)       
    print("Guessed {} of {} defense values.".format(
        estimated, len(DEFENSE_PER_PLAY)))
    s = sorted([(v, k) for k, v in DEFENSE_PER_PLAY.items()])
    print("25 least risky plays:")
    for i in range(25):
        print("{}: {}".format(s[i][1], s[i][0]))
    print("25 riskiest plays")
    for i in range(-25, 0, 1):
        print("{}: {}".format(s[i][1], s[i][0]))
    f = open("defense_per_play_values.txt", "w")
    for k in sorted(DEFENSE_PER_PLAY.keys()):
        f.write("{}\t{}\n".format(k, DEFENSE_PER_PLAY[k]))

# Check the validity of / calculate the score of the new word (if any) formed
# in a lane. 
def score_lane(board, edit_positions, lane_num, is_across):
    dr, dc = (0, 1) if is_across else (1, 0)
    rr, cc = lane_num * dc, lane_num * dr
    w, score, multiplier, has_new = '', 0, 1, False
    while rr <= BOARD_SIZE and cc <= BOARD_SIZE:
        l = False
        if rr != BOARD_SIZE and cc != BOARD_SIZE:
            l = board[rr][cc]
        if l: # still building a word
            w += l
            if (rr, cc) in edit_positions:
                has_new = True
                m = PREMIUMS[rr][cc] # Only new tiles trigger premiums.
                if m == 1: # double word score
                    multiplier *= 2
                score += VALUES[l] * (m if m > 1 else 1)
            else:
                score += VALUES[l]
        elif w:  # reached the end of a word, process it
            if has_new and len(w) >= 2:
                if w not in WORDS:
                    return False, -1
                else:
                    return w, score * multiplier
            w, score, multiplier, has_new = '', 0, 1, False
        rr, cc = rr + dr, cc + dc
    return False, 0

def test_score_lane():
    ___ = False # just to make boards easier to parse visually
    b = [['P', 'H', 'O', 'N', 'E', 'T', 'I', 'C', 'S'],
         [___, ___, ___, 'A', ___, 'A', ___, ___, 'O'],
         [___, ___, 'Q', 'I', ___, 'B', 'Y', 'E', 'S'],
         [___, ___, 'K', 'V', ___, 'L', ___, ___, ___],
         [___, ___, ___, 'E', 'M', 'E', ___, ___, ___],
         [___, ___, 'O', 'R', ___, 'T', 'A', 'W', ___],
         [___, ___, 'X', ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___]]
    for edit_pos, score in ((((0, 0), (0, 4), (0, 8)), 68), # DW-DL-DW
                            (((0, 0), (0, 4)), 34),
                            (((0, 0), (0, 8)), 64),
                            (((0, 4), (0, 8)), 34),
                            (((0, 0),), 32),
                            (((0, 4),), 17),
                            (((0, 8),), 32),
                            (((0, 5), (0, 6), (0, 7), (0, 8)), 32)):
        assert score_lane(b, edit_pos, 0, True) == ('PHONETICS', score)
    assert score_lane(b, ((0, 8),), 8, False) == ('SOS', 6)
    for edit_pos, score in ((((2, 6), (2, 7)), 17),
                            (((2, 6), (2, 7), (2, 8)), 17),
                            (((2, 7), (2, 8)), 9),
                            (((2, 6),), 17),
                            (((2, 7),), 9),
                            (((2, 8),), 9)):
        assert score_lane(b, edit_pos, 2, True) == ('BYES', score)
    assert score_lane(b, ((1, 5), (2, 5), (3, 5)), 5, False) == ('TABLET', 9)
    assert score_lane(b, ((1, 5), (2, 5), (3, 5)), 1, True) == (False, 0)
    assert score_lane(b, ((1, 5), (2, 5), (3, 5)), 2, True) == ('BYES', 9)
    assert score_lane(b, ((1, 5), (2, 5), (3, 5)), 3, True) == (False, 0)  
    assert score_lane(b, ((2, 2),), 2, True) == ('QI', 31)
    assert score_lane(b, ((5, 5), (5, 6), (5, 7)), 5, True) == ('TAW', 7)
    assert score_lane(b, ((5, 5), (5, 6), (5, 7)), 5, False) == ('TABLET', 9)    
    assert score_lane(b, ((5, 5), (5, 6), (5, 7)), 6, False) == (False, 0)
    assert score_lane(b, ((3, 2),), 2, False) == (False, -1)
    assert score_lane(b, ((3, 2),), 3, True) == (False, -1)
    b = empty_board()
    b[4] = [False, False, False, 'V', 'E', 'T', 'C', 'H', False]
    assert (score_lane(
        b, ((4, 3), (4, 4), (4, 5), (4, 6), (4, 7)), 4, True) == (
            'VETCH', 26))
    b[4][2] = 'K'
    b[4][8] = 'Y'
    assert (score_lane(
        b, ((4, 2), (4, 3), (4, 6), (4, 7), (4, 8)), 4, True) == (
            'KVETCHY', 26))

# Find all valid plays on this board using any prefix of the tileset used to
# create it.
def score_board(board, edits, is_across):
    valid_plays = []
    total_perp_score = 0
    perp_words = []
    for i in range(len(edits)):
        edit_r, edit_c, ok_to_score, l = edits[i]
        board[edit_r][edit_c] = l
        if not ok_to_score:
            continue
        edit_positions = [(r, c) for r, c, _, _ in edits[0:i+1]]
        w, score = score_lane(board, edit_positions,
                              edit_r if is_across else edit_c, is_across)
        perp_w, perp_score = score_lane(
            board, edit_positions, edit_c if is_across else edit_r,
            not is_across)
        if perp_score == -1: # invalid word formed perpendicularly, stop trying
            break
        if perp_w:
            total_perp_score += perp_score
            perp_words.append(perp_w)
        if w: # A valid new word must be formed in the play lane.
            valid_plays.append(ValidPlay(
                total_perp_score + score + (
                    BINGO_BONUS if i == RACK_SIZE - 1 else 0),
                tuple([(r, c, l) for r, c, _, l in edits[0:i+1]]),
                tuple(sorted(perp_words + [w]))))
    # Restore the board's original state.
    for edit_r, edit_c, _, _ in edits:
        board[edit_r][edit_c] = False
    return tuple(valid_plays)

def test_score_board():
    ___ = False # just to make boards easier to parse visually
    b = [[___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, 'T', 'U', 'R', 'F', ___, ___],
         [___, ___, ___, 'I', ___, ___, 'O', ___, ___],
         [___, ___, ___, ___, ___, ___, 'A', ___, ___],
         [___, ___, ___, ___, 'A', 'R', 'M', ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___]]
    edits = ((2, 2, True, 'P'),)
    assert score_board(b, edits, True) == (
        ValidPlay(10, ((2, 2, 'P'),), ('PI',)),)
    edits = ((3, 1, False, 'S'), (3, 2, False, 'O'), (3, 3, True, 'P'),
             (3, 4, True, 'H'), (3, 5, True, 'S'))
    assert score_board(b, edits, True) == (
        (ValidPlay(16, ((3, 1, 'S'), (3, 2, 'O'), (3, 3, 'P')),
                   ('SOP', 'TIP')),
         ValidPlay(25, ((3, 1, 'S'), (3, 2, 'O'), (3, 3, 'P'), (3, 4, 'H')),
                   ('HA', 'SOPH', 'TIP'))))
    b = empty_board()
    b[4][2:6] = ['A', 'H', 'E', 'A', 'D']
    edits = ((3, 2, True, 'T'), (3, 3, True, 'O'), (3, 4, True, 'R'),
             (3, 5, True, 'T'), (3, 6, True, 'E'))
    # Note that the across play of just T is not considered valid. The
    # down version is the one we want to use.
    assert score_board(b, edits, True) == (
        (ValidPlay(11, ((3, 2, 'T'), (3, 3, 'O')), ('OH', 'TA', 'TO')),
         ValidPlay(14, ((3, 2, 'T'), (3, 3, 'O'), (3, 4, 'R')),
                   ('OH', 'RE', 'TA', 'TOR')),
         ValidPlay(19, ((3, 2, 'T'), (3, 3, 'O'), (3, 4, 'R'), (3, 5, 'T')),
                   ('OH', 'RE', 'TA', 'TA', 'TORT')),
         ValidPlay(43,
                   ((3, 2, 'T'), (3, 3, 'O'), (3, 4, 'R'), (3, 5, 'T'),
                    (3, 6, 'E')),
                   ('ED', 'OH', 'RE', 'TA', 'TA', 'TORTE'))))
    # Make sure we actually get the down version!
    assert ValidPlay(2, ((3, 2, 'T'),), ('TA',)) in score_board(
        b, edits, False)
    edits = ((3, 2, True, 'H'), (3, 3, True, 'O'), (3, 4, True, 'R'),
             (3, 5, True, 'S'), (3, 6, True, 'E'))
    assert score_board(b, edits, True) == (
         ValidPlay(17, ((3, 2, 'H'), (3, 3, 'O')), ('HA', 'HO', 'OH')),)
    b = empty_board()
    edits = ((0, 4, False, 'T'), (1, 4, False, 'O'), (2, 4, False, 'R'),
             (3, 4, False, 'T'), (4, 4, False, 'H'))
    assert score_board(b, edits, False) == tuple()
    edits = ((1, 4, False, 'T'), (2, 4, False, 'O'), (3, 4, False, 'R'),
             (4, 4, True, 'T'), (5, 4, True, 'H'))
    assert score_board(b, edits, False) == (
        (ValidPlay(8, ((1, 4, 'T'), (2, 4, 'O'), (3, 4, 'R'), (4, 4, 'T')),
                   ('TORT',)),))
    b = empty_board()
    b[2][4] = 'V'
    b[3][4] = 'E'
    b[4][4] = 'T'
    edits = ((1, 4, True, 'K'), (5, 4, True, 'C'), (6, 4, True, 'H'),
             (7, 4, True, 'Y'), (8, 4, True, 'Q'))
    assert score_board(b, edits, False) == (
        (ValidPlay(18, ((1, 4, 'K'), (5, 4, 'C'), (6, 4, 'H')),
                   ('KVETCH',)),
         ValidPlay(22, ((1, 4, 'K'), (5, 4, 'C'), (6, 4, 'H'), (7, 4, 'Y')),
                   ('KVETCHY',))))

# Determine where tiles would be played, starting from the given coordinates
# and moving in the given direction. Return info on where each tile would fall
# and at what point the play is connected with the rest of the board.
def check_lane(board, neighbors, r, c, is_across, max_tiles):
    placement_info = []
    connected = False
    rr, cc = r, c
    dr, dc = (0 if is_across else 1, 1 if is_across else 0)
    results = []
    # See how many tiles need to be played for connectedness.
    for i in range(max_tiles):
        while rr < BOARD_SIZE and cc < BOARD_SIZE and board[rr][cc]:
            rr, cc = rr + dr, cc + dc
        if rr == BOARD_SIZE or cc == BOARD_SIZE:
            return tuple(placement_info)
        if neighbors[rr][cc]:
            connected = True
        placement_info.append((rr, cc, connected))
        # pretend to place a tile
        rr, cc = rr + dr, cc + dc
    return tuple(placement_info)

def test_check_lane():
    ___ = False # just to make boards easier to parse visually
    b = empty_board()
    n = neighboring_cells(b)
    assert check_lane(b, n, 0, 4, False, 5) == (
        (0, 4, False), (1, 4, False), (2, 4, False), (3, 4, False),
        (4, 4, True))
    assert check_lane(b, n, 4, 0, True, 4) == (
        (4, 0, False), (4, 1, False), (4, 2, False), (4, 3, False))
    assert check_lane(b, n, 4, 4, True, 1) == ((4, 4, True),)
    assert check_lane(b, n, 5, 2, True, 5) == (
        (5, 2, False), (5, 3, False), (5, 4, False), (5, 5, False),
        (5, 6, False))
    b = [[___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, 'A', ___, ___, ___],
         [___, 'A', ___, 'A', ___, 'A', ___, 'A', ___],
         [___, 'A', 'A', 'A', 'A', 'A', 'A', 'A', ___],
         [___, ___, ___, ___, 'A', ___, ___, ___, ___],
         [___, ___, ___, 'A', 'A', ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___],
         [___, ___, ___, ___, ___, ___, ___, ___, ___]]
    n = neighboring_cells(b)
    assert check_lane(b, n, 1, 2, True, 5) == (
        (1, 2, False), (1, 3, False), (1, 4, False), (1, 5, True), (1, 6, True))
    assert check_lane(b, n, 3, 0, True, 5) == (
        (3, 0, True), (3, 2, True), (3, 4, True), (3, 6, True), (3, 8, True))
    assert check_lane(b, n, 5, 8, False, 5) == (
        (5, 8, False), (6, 8, False), (7, 8, False), (8, 8, False))
    assert check_lane(b, n, 2, 3, False, 5) == (
        (2, 3, True), (5, 3, True), (7, 3, True), (8, 3, True))
    assert check_lane(b, n, 5, 3, True, 1) == ((5, 3, True),)
    assert check_lane(b, n, 0, 0, False, 3) == (
        (0, 0, False), (1, 0, False), (2, 0, False))

# Find and return coordinates of all cells next to a played tile.
def neighboring_cells(board):
    neighbors = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    if not board[BOARD_SIZE // 2][BOARD_SIZE // 2]:
        neighbors[BOARD_SIZE // 2][BOARD_SIZE // 2] = True
    else:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c]:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        rr, cc = r + dr, c + dc
                        if (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                            and not board[rr][cc]):
                            neighbors[rr][cc] = True
    return neighbors

def test_neighboring_cells():
    ___ = False # just to make boards easier to parse visually
    TTT = True
    b = empty_board()
    n = neighboring_cells(b)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) == (BOARD_SIZE // 2, BOARD_SIZE // 2):
                assert n[r][c]
            else:
                assert not n[r][c]
    b[3][4] = ['A']
    b[4][4] = ['A']
    assert neighboring_cells(b) == [
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, TTT, ___, ___, ___, ___],
        [___, ___, ___, TTT, ___, TTT, ___, ___, ___],
        [___, ___, ___, TTT, ___, TTT, ___, ___, ___],
        [___, ___, ___, ___, TTT, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___]]
    b[4][3], b[4][5], b[5][4] = 'A', 'A', 'A'
    assert neighboring_cells(b) == [
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, TTT, ___, ___, ___, ___],
        [___, ___, ___, TTT, ___, TTT, ___, ___, ___],
        [___, ___, TTT, ___, ___, ___, TTT, ___, ___],
        [___, ___, ___, TTT, ___, TTT, ___, ___, ___],
        [___, ___, ___, ___, TTT, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___],
        [___, ___, ___, ___, ___, ___, ___, ___, ___]]
    b = [
        [___, ___, ___, ___, 'A', 'A', 'A', 'A', 'A'],
        [___, 'A', 'A', 'A', 'A', ___, ___, ___, 'A'],
        [___, 'A', ___, ___, 'A', ___, ___, ___, ___],
        [___, 'A', ___, ___, 'A', 'A', ___, ___, ___],
        [___, 'A', 'A', 'A', 'A', 'A', 'A', 'A', ___],
        [___, ___, 'A', ___, ___, ___, 'A', ___, ___],
        [___, ___, 'A', ___, ___, ___, 'A', 'A', 'A'],
        ['A', 'A', 'A', ___, ___, ___, 'A', ___, ___],
        [___, ___, 'A', 'A', ___, 'A', 'A', ___, ___]]
    assert neighboring_cells(b) == [
        [___, TTT, TTT, TTT, ___, ___, ___, ___, ___],
        [TTT, ___, ___, ___, ___, TTT, TTT, TTT, ___],
        [TTT, ___, TTT, TTT, ___, TTT, ___, ___, TTT],
        [TTT, ___, TTT, TTT, ___, ___, TTT, TTT, ___],
        [TTT, ___, ___, ___, ___, ___, ___, ___, TTT],
        [___, TTT, ___, TTT, TTT, TTT, ___, TTT, TTT],
        [TTT, TTT, ___, TTT, ___, TTT, ___, ___, ___],
        [___, ___, ___, TTT, ___, TTT, ___, TTT, TTT],
        [TTT, TTT, ___, ___, TTT, ___, ___, TTT, ___]]

# Find and return all valid plays for a given rack on a given board.
def find_valid_plays(board, rack):
    valid_plays = set()
    neighbors = neighboring_cells(board)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                continue
            for is_across in (True, False):
                placement_info = check_lane(board, neighbors, r, c, is_across,
                                            len(rack))
                # TODO: Instead of feeding every permutation in, do some clever
                # backtracking within score_board itself. This would avoid some
                # redundant work.
                for p in set(itertools.permutations(rack)):
                    edits = []
                    for i in range(len(placement_info)):
                        rr, cc, connected = placement_info[i]
                        edits.append((rr, cc, connected, p[i]))
                    vps = score_board(board, edits, is_across)
                    for vp in vps:
                        valid_plays.add(vp)
    return valid_plays

# Light overall functionality test since the innards are well-unit tested.
def test_find_valid_plays():
    ___ = False # just to make boards easier to parse visually
    b = [[___, ___, ___, ___, ___, ___, 'F', 'O', 'X'],
         [___, ___, ___, ___, ___, ___, 'O', 'R', ___],
         [___, ___, ___, ___, ___, 'P', 'E', 'T', ___],
         [___, ___, ___, ___, 'J', 'O', ___, ___, ___],
         [___, ___, ___, ___, 'A', 'I', 'T', ___, 'H'],
         [___, ___, ___, ___, 'B', ___, 'S', ___, 'U'],
         [___, ___, 'M', 'A', 'S', ___, 'K', 'A', 'E'],
         [___, 'Q', 'I', ___, ___, ___, ___, 'G', ___],
         ['A', 'I', 'L', ___, ___, ___, 'C', 'E', 'E']]
    assert not find_valid_plays(b, 'NNYZ')
    vp = find_valid_plays(b, 'DRVW')
    assert len(vp) == 1
    assert list(vp)[0] == ValidPlay(7, ((4, 3, 'W'),), ('WAIT',))
    b = empty_board()
    assert not find_valid_plays(b, 'CHKVY')
    b[4][4:5] = ['E', 'T']
    vps = find_valid_plays(b, 'CHKVY')
    assert ValidPlay(
        46,
         ((4, 2, 'K'), (4, 3, 'V'), (4, 6, 'C'), (4, 7, 'H'), (4, 8, 'Y')),
         ('KVETCHY',)) in vps
    assert ValidPlay(
        12,
         ((3, 4, 'Y'), (5, 4, 'C'), (6, 4, 'H')),
         ('YECH',)) in vps
    b = empty_board()
    b[4][2:6] = ['A', 'H', 'E', 'A', 'D']
    vps = find_valid_plays(b, 'EORTT')
    assert ValidPlay(
        43,
        ((3, 2, 'T'), (3, 3, 'O'), (3, 4, 'R'), (3, 5, 'T'), (3, 6, 'E')),
        ('ED', 'OH', 'RE', 'TA', 'TA', 'TORTE')) in vps
    assert ValidPlay(
        43,
        ((5, 2, 'T'), (5, 3, 'O'), (5, 4, 'R'), (5, 5, 'T'), (5, 6, 'E')),
        ('AT', 'AT', 'DE', 'ER', 'HO', 'TORTE')) in vps

# Find and return all exchanges from a given rack.
# Put the largest exchanges last so that strategies can take advantage of that.
def find_exchanges(rack):
    exch = []
    for n in range(1, RACK_SIZE + 1):
        seen = set()
        for c in itertools.combinations(rack, n):
            seen.add(tuple(sorted(c)))
        exch.extend([''.join(s) for s in seen])
    return exch

def test_find_exchanges():
    r = []
    assert find_exchanges(r) == []
    r = ['A']
    assert find_exchanges(r) == ['A']
    r = ['A', 'A', 'B']
    e = find_exchanges(r)
    assert len(e) == 5
    for ex in ['A', 'B', 'AA', 'AB', 'AAB']:
        assert ex in e
    r = ['A', 'B', 'C', 'D', 'E']
    e = find_exchanges(r)
    assert e[-1] == 'ABCDE'
    assert len(e) == 31

def draw(bag, rack):
    letters_drawn = []
    while bag and len(rack) < RACK_SIZE:
        letter = bag.pop()
        letters_drawn.append(letter)
        rack.append(letter)
    return letters_drawn

def exchange(bag, rack, tiles):
    assert len(bag) >= len(tiles)
    exchanged = []
    for l in tiles:
        exchanged.append(rack.pop(rack.index(l)))
    draw(bag, rack)
    bag += exchanged # only add these after drawing new tiles
    random.shuffle(bag)

def remove_played_tiles(rack, to_remove):
    for l in to_remove:
        rack.remove(l)

def empty_board():
    return [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def rep_board(board):
    return '\n'.join([''.join([c if c else '.' for c in l]) for l in board])

def rep_rack(rack):
    return ''.join(sorted(rack))

def rep_words(scored_words):
    return ','.join(sorted(scored_words))

# ********** BEGIN STRATEGIES **********

# A strategy takes the following inputs:
# valid_plays: set of all valid plays, each as a ValidPlay tuple
# board: the current board
# rack: the active player's rack
# unseen: the remaining contents of the bag + opponent's rack (sorted)
# TODO: will probably need more inputs like current scores
# TODO: maybe better to pass around game state objects?
# TODO: these strategies themselves should probably also be objects

# Choose a valid play uniformly at random.
def random_strat(valid_plays, valid_exchanges, board, rack, unseen,
                 tiles_in_bag, bag):
    if valid_plays:
        return PLAY, random.choice(list(valid_plays))
    elif valid_exchanges:
        return EXCHANGE, random.choice(list(valid_exchanges))
    else:
        return PASS, None

# Choose a play with the highest score.
def greedy_strat(valid_plays, valid_exchanges, board, rack, unseen,
                 tiles_in_bag, bag):
    if valid_plays:
        return PLAY, sorted(list(valid_plays))[-1] # one of highest-scoring
    elif valid_exchanges:
        return EXCHANGE, sorted(list(valid_exchanges))[-1] # one of biggest
    else:
        return PASS, None
    
    return PLAY, valid_plays[0]

# Adjust the score of each potential move based on data from two future moves.
# The 0.4 parameter comes from experimentation.
def leave_strat(valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, bag, m=0.4):
    best_adj_score = -1000
    best_choice = PASS, None
    for v in valid_plays:
        r_copy = rack[:]
        remove_played_tiles(r_copy, [t for _, _, t in v.edits])
        adj_score = v.score + m * LEAVES[''.join(sorted(r_copy))]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = PLAY, v
    for ex in valid_exchanges:
        leave = rack[:]
        remove_played_tiles(leave, ex)
        adj_score = m * LEAVES[''.join(sorted(leave))]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = EXCHANGE, ex
    return best_choice

# Like leave_strat, but with an adjustable m parameter.
def leave_strat_m(m):
    return (lambda valid_plays, valid_exchanges, board, rack, unseen,
            tiles_in_bag : leave_strat(
                valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, bag, m=m), bag)

def defense_strat(valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, bag, m=0.1, tile_threshold=20):
    # This strategy is likely to become less reliable later in the game, since
    # the board layouts get more and more specific and so the general data fits
    # less well. So, cut off the strategy once the game progresses past a
    # certain point.
    if tiles_in_bag < tile_threshold:
        return greedy_strat(valid_plays, valid_exchanges, board, rack,
                            unseen, tiles_in_bag, None)
    best_adj_score = -1000
    best_play = None
    for v in valid_plays:
        played_locations_cv = tuple(
            [(1 if l in VOWELS else 0, r, c) for r, c, l in v.edits])
        # These values represent risk / opponent score, so subtract them.
        adj_score = v.score - m * DEFENSE_PER_PLAY[played_locations_cv]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_play = v
    if best_play:
        return PLAY, best_play
    if valid_exchanges:
        return EXCHANGE, sorted(list(valid_exchanges))[-1] # one of biggest
    return PASS, None

def defense_strat_mt(m, t):
    return (lambda valid_plays, valid_exchanges, board, rack, unseen,
            tiles_in_bag, bag: defense_strat(
                valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, bag, m=m, tile_threshold=t))

# Combine Leave and Defense.
def leave_defense_strat(valid_plays, valid_exchanges, board, rack, unseen,
                  tiles_in_bag):
    m = 0.3
    best_adj_score = -1000
    best_choice = PASS, None
    for v in valid_plays:
        r_copy = rack[:]
        remove_played_tiles(r_copy, [t for _, _, t in v.edits])
        adj_score = v.score + m * LEAVES[''.join(sorted(r_copy))]
        if tiles_in_bag >= 20:
            played_locations_cv = tuple(
                [(1 if l in VOWELS else 0, r, c) for r, c, l in v.edits])
            adj_score -= m * DEFENSE_PER_PLAY[played_locations_cv]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = PLAY, v
    for ex in valid_exchanges:
        leave = rack[:]
        remove_played_tiles(leave, ex)
        adj_score = m * LEAVES[''.join(sorted(leave))]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = EXCHANGE, ex
    return best_choice  

# Assume a greedy opponent, see what they might do next turn in response, and
# decide accordingly.
def lookahead_1_strat(valid_plays, valid_exchanges, board, rack, unseen,
                      tiles_in_bag, bag, num_trials=10, num_candidates=10):
    if not valid_plays:
        if valid_exchanges:
            return EXCHANGE, valid_exchanges[-1]
        else:
            return PASS, None
    best_delta = -1000
    best_play = None
    num_candidates = min(len(valid_plays), num_candidates)
    candidates = sorted(list(valid_plays))[-num_candidates:]
    # Use the same set of "next tiles" for each play, to compare on equal terms.
    opp_scores = [0]*num_candidates
    for _ in range(num_trials):
        next_tiles = random.sample(unseen, min(len(unseen), 10))
        for i in range(len(candidates)):
            edits = candidates[i].edits
            for rr, cc, l in edits:
                board[rr][cc] = l
            opp_rack = next_tiles[
                len(edits) : min(len(edits) + RACK_SIZE, len(next_tiles)-1)]
            opp_valid_exchanges = (
                find_exchanges(opp_rack)
                if tiles_in_bag - len(edits) - len(opp_rack) >= RACK_SIZE
                else [])
            opp_valid_plays = find_valid_plays(board, opp_rack)
            # Skip the last four inputs because greedy_strat doesn't use them.
            choice, details = greedy_strat(
                opp_valid_plays, opp_valid_exchanges, None, None, None, None, None)
            if choice == PLAY:
                opp_scores[i] += details[0]
            for rr, cc, _ in edits:
                board[rr][cc] = False
    for i in range(len(candidates)):
        delta = candidates[i].score - (opp_scores[i] / num_trials)
        if delta > best_delta:
            best_delta = delta
            best_play = candidates[i]
    return PLAY, best_play

def find_next_round_scores(candidate_unseen, candidate, board, tiles_in_bag, bag, rack):
    next_turn_score = 0
    opp_score = 0

    #Update board for candidate play and remove played letters from rack
    edits = candidate.edits
    for rr, cc, l in edits:
        rack.remove(l)
        board[rr][cc] = l

    letters_drawn = draw(bag, rack)
    for letter in letters_drawn:
        candidate_unseen.remove(letter)

    opp_rack_sim = random.sample(candidate_unseen, min(len(candidate_unseen), RACK_SIZE))
    #Simulate opponent's turn
    opp_valid_exchanges = (
        find_exchanges(opp_rack_sim)
        if tiles_in_bag - len(edits) - len(opp_rack_sim) >= RACK_SIZE
        else [])
    opp_valid_plays = find_valid_plays(board, opp_rack_sim)
    # Skip the last four inputs because greedy_strat doesn't use them.
    choice, details = greedy_strat(
        opp_valid_plays, opp_valid_exchanges, None, None, None, None, None)

    if choice == PLAY:
        opp_score += details[0]
        opp_edits = details[1]
        for rr, cc, l in opp_edits:
            candidate_unseen.remove(l)
            if l in bag:
                bag.remove(l)
            board[rr][cc] = l

    #Simulate player next turn
    next_turn_valid_exchanges = (
        find_exchanges(rack)
        if tiles_in_bag - len(edits) - len(rack) >= RACK_SIZE
        else [])
    player_next_turn_valid_plays = find_valid_plays(board, rack)
    # Skip the last four inputs because greedy_strat doesn't use them.
    next_turn_choice, next_turn_details = greedy_strat(
        player_next_turn_valid_plays, next_turn_valid_exchanges, None, None, None, None, None)

    if next_turn_choice == PLAY:
        next_turn_score += next_turn_details[0]

    return opp_score, next_turn_score

#def find_next_turn_sample():
def score_candidate(num_candidates, unseen, candidate, board, tiles_in_bag, bag, rack, n, level=1):
    candidate_board = copy.deepcopy(board)
    player_rack = copy.deepcopy(rack)
    candidate_unseen = copy.deepcopy(unseen)
    candidate_bag = copy.deepcopy(bag)

    #Find candidate score
    opp_score, next_turn_score = \
        find_next_round_scores(candidate_unseen, candidate, candidate_board, tiles_in_bag, candidate_bag, player_rack)
    candidate_score = candidate.score - opp_score + next_turn_score

    #Find sum of sub-tree scores
    level_sum = 0.0
    if n != level:
        new_candidates = find_new_candidates(candidate_board, player_rack, num_candidates)
        if new_candidates != []:
            for i in range(num_candidates):
                k = i
                if len(new_candidates) == 1:
                    k = 0
                cand_score = score_candidate(num_candidates, candidate_unseen, new_candidates[k], candidate_board, tiles_in_bag, candidate_bag, player_rack, n, level + 1)
                level_sum += cand_score
    #Average sub-tree scores
    level_score = level_sum/float(num_candidates)
    candidate_score += level_score
    return candidate_score

def score_candidates(candidates, num_candidates, num_trials, unseen, board, tiles_in_bag, n, rack, bag):
    #TO DO: implement multiple trials
    candidate_scores = [0]*num_candidates
    if len(candidates) >= num_candidates:
        candidate_scores = [candidates[i].score for i in range(num_candidates)]
    for i in range(num_candidates):
        if i < len(candidates):
            candidate_scores[i] += score_candidate(num_candidates, unseen, candidates[i], board, tiles_in_bag, bag, rack, n, level=1)
    return candidate_scores

def find_best_play(candidates, candidate_scores):
    top_expected_score = max(candidate_scores)
    best_play_index = candidate_scores.index(top_expected_score)
    best_play = candidates[best_play_index]
    return best_play

def find_new_candidates(board, rack, num_candidates):
    valid_plays = find_valid_plays(board, rack)
    num_candidates = min(len(valid_plays), num_candidates)
    candidates = sorted(list(valid_plays))[-num_candidates:]  # contains duplicate words played in different places
    return candidates

def lookahead_n_strat(valid_plays, valid_exchanges, board, rack, unseen,
                      tiles_in_bag, bag, n = 1, num_trials=10, num_candidates=10):
    candidates = find_new_candidates(board, rack, num_candidates)
    if tiles_in_bag > RACK_SIZE:
        candidate_totals = [0]*num_candidates
        for i in range(num_trials):
            if not valid_plays:
                valid_exchanges = (find_exchanges(rack) if tiles_in_bag >= RACK_SIZE else [])
                if valid_exchanges:
                    return EXCHANGE, valid_exchanges[-1]
                else:
                    return PASS, None
            candidate_scores = score_candidates(candidates, num_candidates, num_trials, unseen, board, tiles_in_bag, n, rack, bag)
            candidate_totals = [candidate_totals[i] + candidate_scores[i] for j in range(num_candidates)]
        best_play = find_best_play(candidates, candidate_totals)
        return PLAY, best_play
    else:
        return endgame_strat(valid_plays, valid_exchanges, board, rack, unseen, tiles_in_bag, bag)

#def general_strat(look_ahead_steps, num_simulations, next_turn_weight, board_score_weight,
#                  sabotage_weight, simulation_depth)

def endgame_strat(valid_plays, valid_exchanges, board, rack, unseen,
                  tiles_in_bag, bag):
    if len(unseen) <= 10:
        return lookahead_1_strat(valid_plays, valid_exchanges, board, rack,
                                 unseen, tiles_in_bag, bag)
    else:
        return greedy_strat(valid_plays, valid_exchanges, board, rack, unseen,
                            tiles_in_bag, bag)

# ********** END STRATEGIES **********

def sim(strat1, strat2, log=True):
    strats = [strat1, strat2]
    scores = [0, 0]  
    record = [[], []]
    board = empty_board()
    bag = []
    for l, f in FREQS.items(): # Initialize bag
        for i in range(f):
            bag.append(l)
    random.shuffle(bag)
    racks = [[], []]
    draw(bag, racks[0])
    draw(bag, racks[1])
    active = 0  # 0 on first player's turn, 1 on second player's turn
    if log:
        print("\nGAME START!\n")
    wordless_turns = 0 # the game ends automatically after 6 wordless turns
    while racks[0] and racks[1]: # or when a rack is empty
        valid_plays = find_valid_plays(board, racks[active])
        valid_exchanges = (
            find_exchanges(racks[active]) if len(bag) >= RACK_SIZE else [])
        choice, details = strats[active](valid_plays, valid_exchanges, board, racks[active],
                                         sorted(racks[1-active] + bag), len(bag), bag)
        if choice == PLAY:
            score, edits, scored_words = details
            for rr, cc, l in edits:
                board[rr][cc] = l
            old_rack = racks[active][:]
            remove_played_tiles(racks[active], [l for _, _, l in edits])
            scores[active] += score
            scoreline = "rack {}, played r{}c{} {}  score {}".format(
                rep_rack(old_rack), edits[0][0]+1, edits[0][1]+1,
                rep_words(scored_words), score)
            record[active].append((PLAY, score, edits, rep_rack(racks[active])))
            draw(bag, racks[active])
            wordless_turns = 0
        elif choice == EXCHANGE:
            wordless_turns += 1
            assert bag
            oldrack = racks[active][:]
            exchange(bag, racks[active], details)
            scoreline = "rack {}, exchanged {}, redrew to {}".format(
                rep_rack(oldrack), details, rep_rack(racks[active]))
            record[active].append((EXCHANGE, 0, None, rep_rack(racks[active])))
        else: # PASS
            wordless_turns += 1
            scoreline = "rack {}, passed".format(rep_rack(racks[active]))
            record[active].append((PASS, 0, None, rep_rack(racks[active])))
        if log:
            print("{}-{}   P{} {}".format(
                scores[0], scores[1], active+1, scoreline))
            if choice == PLAY:
                print(rep_board(board))
            print("")
        if wordless_turns == 6:
            break
        active = 1 - active # switch active player

    # Handle penalties for remaining tiles in racks.
    for p in range(2):
        if racks[p]:
            penalty = sum([VALUES[l] for l in racks[p]])
            scores[p] -= penalty
            if log:
                print("{}-{}   P{} left {} penalty -{}".format(
                    scores[0], scores[1], p+1, rep_rack(racks[p]), penalty))
    return scores, record

# Play two strategies against each other for many trials, and report the
# results (and an approximate sense of statistical significance).
# Alternate who goes first, because 
def compare_strats(strat1, strat2, num_trials, log_each_game=False,
                   progress_update_every=100):
    assert num_trials % 2 == 0, "Number of trials must be even"
    wins = [0, 0]
    score_totals = [0, 0]
    strats = [strat1, strat2]
    for i in range(num_trials):
        goes_first = i % 2
        scores, _ = sim(strats[goes_first], strats[1-goes_first],
                        log=log_each_game)
        if goes_first == 0:
            score1, score2 = scores
        else:
            score2, score1 = scores
        if i % progress_update_every == 0:
            print("Game {} ({} goes first): P1 {} P2 {}".format(
                i+1, goes_first+1, score1, score2))
        if score1 > score2:
            wins[0] += 1
        elif score2 > score1:
            wins[1] += 1
        else:
            wins[0] += 0.5
            wins[1] += 0.5
        score_totals[0] += score1
        score_totals[1] += score2
    print(
        "P1 ({}) won {} (avg. score {})\nP2 ({}) won {} (avg.score {})".format(
            strat1.__name__, wins[0], round(score_totals[0] / num_trials, 1),
            strat2.__name__, wins[1], round(score_totals[1] / num_trials, 1)))
    # Use a normal approximation to the binomial distribution.
    if num_trials >= 1:#wins[0] > wins[1] and
        print(
            "Prob. of at least this big a positive difference "
            "for equal strategies: {}".format(
                math.erfc((wins[0] - num_trials * 0.5)/(0.25*num_trials)**0.5)))
    return wins

def compile_leave_and_defense_data(num_trials, log_every=100):
    per_leave_data = {}
    per_cell_counts = empty_defense_stats()
    per_cell_totals = empty_defense_stats()
    per_play_data = {}
    for t in range(num_trials):
        if t % log_every == 0:
            print(t)
        _, record = sim(greedy_strat, greedy_strat, log=False)
        # LEAVE DATA
        # Infer which leaves are associated with higher scores for us on the
        # next two turns.
        for i in range(2):
            for j in range(len(record[i])-2):
                _, _, _, leave = record[i][j]
                if len(leave) == RACK_SIZE:
                    continue
                move_1, score_1, _, _ = record[i][j+1]
                move_2, score_2, _, _ = record[i][j+2]
                # Don't bias too heavily toward the endgame.
                if move_1 != PLAY and move_2 != PLAY:
                    continue
                score = score_1 + score_2
                if leave not in per_leave_data:
                    per_leave_data[leave] = [0, 0]
                per_leave_data[leave][0] += score
                per_leave_data[leave][1] += 1
        # DEFENSE DATA
        # Infer which play positions and board squares are associated with
        # higher opponent scores in response. Keep track of consonants and
        # vowels separately. (Putting vowels next to DL/TL premium squares is
        # bad because of responses like EX, JO, QI, ZA, etc.)
        for i in range(2):
            for j in range(len(record[i])):
                opp_next_move_index = j + i
                if opp_next_move_index > len(record[1-i]) - 1:
                    continue
                move, _, edits, _ = record[i][j]
                if move != PLAY:
                    continue
                _, opp_score, _, _ = record[1-i][opp_next_move_index]
                edits_cv = tuple([
                    (1 if l in VOWELS else 0, r, c) for r, c, l in edits])
                for v, r, c in edits_cv:
                    per_cell_counts[v][r][c] += 1
                    per_cell_totals[v][r][c] += opp_score               
                if edits_cv not in per_play_data:
                    per_play_data[edits_cv] = [0, 0]
                per_play_data[edits_cv][0] += opp_score
                per_play_data[edits_cv][1] += 1
    f = open("leaves.txt", "w")
    for k, v in sorted(per_leave_data.items()):
        total, instances = v
        f.write("{},{},{}\n".format(k, total, instances))
    f = open("defense_per_cell.txt", "w")
    f.write("{}\n{}".format(per_cell_totals, per_cell_counts))
    f = open("defense_per_play.txt", "w")
    for k, v in sorted(per_play_data.items()):
        total, instances = v
        f.write("{}\t{}\t{}\n".format(k, total, instances))

### BEGIN MAIN BODY ###
if RUN_TESTS:
    test_score_lane()
    test_score_board()
    test_check_lane()
    test_neighboring_cells()
    test_find_valid_plays()
    test_find_exchanges()

#compile_leave_and_defense_data(250000)
#sim(random_strat, leave_strat, log=True)
sim(greedy_strat, lookahead_n_strat, log=True)
#compare_strats(greedy_strat, lookahead_n_strat, 50)
#sim(defense_strat_mt(0.15, 25), lookahead_n_strat)
#compare_strats(defense_strat_mt(0.15, 25), lookahead_n_strat(), 2)
"""
for t in range(10, 35, 5):
    for i in range(10, 110, 10):
        print(t, i*0.01)
        compare_strats(defense_strat_mt(i*0.01, t), greedy_strat, 5000,
                       progress_update_every=1000)
"""