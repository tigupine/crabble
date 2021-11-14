from collections import namedtuple
import itertools
import random
from string import ascii_lowercase

ValidPlay = namedtuple(
    'ValidPlay', ['score', 'r', 'c', 'p', 'edits', 'scored_words'])

RACK_SIZE = 5
# 1 = double word, 2 = double letter, 3 = triple letter
BOARD = [[1,0,0,0,2,0,0,0,1],
         [0,1,0,0,0,0,0,1,0],
         [0,0,3,0,0,0,3,0,0],
         [0,0,0,2,0,2,0,0,0],
         [2,0,0,0,1,0,0,0,2],
         [0,0,0,2,0,2,0,0,0],
         [0,0,3,0,0,0,3,0,0],
         [0,1,0,0,0,0,0,1,0],
         [1,0,0,0,2,0,0,0,1]]

BOARD_SIZE = len(BOARD)

VALUES = {'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4,
          'G': 2, 'H': 4, 'I': 1, 'J': 8, 'K': 5, 'L': 1,
          'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
          'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
          'Y': 4, 'Z': 10, '?': 0}

# 40 tiles
FREQS = {'A': 3, 'B': 1, 'C': 1, 'D': 1, 'E': 4, 'F': 1,
         'G': 1, 'H': 1, 'I': 3, 'J': 1, 'K': 1, 'L': 1,
         'M': 1, 'N': 2, 'O': 3, 'P': 1, 'Q': 1, 'R': 2,
         'S': 2, 'T': 2, 'U': 1, 'V': 1, 'W': 1, 'X': 1,
         'Y': 1, 'Z': 1, '?': 1}

# Bonus for using entire rack
BINGO_BONUS = 15

f = open("words.txt", "r")
WORDS = set([l.strip() for l in f.readlines() if len(l) <= BOARD_SIZE + 1])

# Options for strategies to return.
PLAY = 'PLAY'
EXCHANGE = 'EXCHANGE'
PASS = 'PASS'

# TODO: write tests for these functions

def check_board(board, edits, across, r, c, first_play):
    def value(l):
        return VALUES.get(l, 0)
    checks = set()
    if across:
        checks.add((r, True))
        for _, c, _ in edits:
            checks.add((c, False))
    else:
        checks.add((c, False))
        for r, _, _ in edits:
            checks.add((r, True))
    total_score = 0
    scored_words = []
    connected = False
    edit_positions = set([(r, c) for r, c, _ in edits])
    for i, across in checks:
        dr, dc = (0, 1) if across else (1, 0)
        w, score, multiplier, has_old, has_new = '', 0, 1, False, False
        rr, cc = i * dc, i * dr
        while rr <= BOARD_SIZE and cc <= BOARD_SIZE:
            l = False
            if rr != BOARD_SIZE and cc != BOARD_SIZE:
                l = board[rr][cc]
            if l:
                w += l
                if (rr, cc) in edit_positions:
                    has_new = True
                    m = BOARD[rr][cc]
                    if m == 1:
                        multiplier *= 2
                        score += value(l)
                    else:
                        score += value(l) * (m if m != 0 else 1)
                else:
                    has_old = True
                    score += value(l)
            elif w:  # reached the end of a word, process it
                if len(w) >= 2:
                    if has_old and has_new:
                        connected = True
                    if w.upper() not in WORDS:
                        return False, None
                    if has_new:
                        total_score += score * multiplier
                        if len(edits) == RACK_SIZE:
                            total_score += BINGO_BONUS
                        scored_words.append(w)
                w, score, multiplier, has_old, has_new = (
                 '', 0, 1, False, False)
            rr, cc = rr + dr, cc + dc
    # Only the first play of the game is allowed to be disconnected.
    if not connected and not first_play:
        return False, None
    return total_score, scored_words

def check_play(board, r, c, d, p, first_play):
    def undo_edits(board, edits):
        for rr, cc, _ in edits:
            board[rr][cc] = False
    edits = []
    new_pos = []
    rr, cc = r, c
    dr, dc = d
    # Try to play all the tiles.
    for i in range(len(p)):
        while rr < BOARD_SIZE and cc < BOARD_SIZE and board[rr][cc]:
            rr, cc = rr + dr, cc + dc
        if rr == BOARD_SIZE or cc == BOARD_SIZE:
            undo_edits(board, edits)
            return False, None, None
        board[rr][cc] = p[i]
        edits.append((rr, cc, p[i]))
    if first_play and not board[BOARD_SIZE//2][BOARD_SIZE//2]:
        undo_edits(board, edits)
        return False, None, None
    score, scored_words = check_board(
        board, edits, (dc == 1), r, c, first_play)
    undo_edits(board, edits)
    return score, edits, scored_words
    
def find_valid_plays(board, rack, first_play):
    plays = set()
    valid_plays = []
    for n in range(1, RACK_SIZE + 1):
        for c in itertools.combinations(rack, n):
            for p in itertools.permutations(c):
                if '?' in p:
                    blank_index = p.index('?')
                    for l in ascii_lowercase:
                        plays.add(p[0:blank_index] + (l,) + p[blank_index+1:])
                else:
                    plays.add(p)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                continue
            for d in ((0, 1), (1, 0)):
                for p in plays:
                    score, edits, scored_words = check_play(
                        board, r, c, d, p, first_play)
                    # Since we only have 1 blank, it is impossible to score 0.
                    if score:
                        valid_plays.append(
                            ValidPlay(score, r, c, p, edits, scored_words))
    return valid_plays

def draw(bag, rack):
    while bag and len(rack) < RACK_SIZE:
        rack.append(bag.pop())

def exchange(bag, rack, tiles):
    exchanged = []
    for l in tiles:
        exchanged.append(rack.pop(rack.index(l)))
    draw(bag, rack)
    bag += exchanged
    random.shuffle(bag)
    # Draw up if that's still needed.
    draw(bag, rack)

def rep_board(board):
    return '\n'.join([''.join([c if c else '.' for c in l]) for l in board])

def rep_rack(rack):
    return ''.join(rack)

def rep_words(scored_words):
    return ','.join(scored_words) 

# ********** BEGIN STRATEGIES **********

# A strategy takes the following inputs:
# valid_plays: list of all valid plays, each as a ValidPlay tuple
# board: the current board
# rack: the active player's rack
# bag: the remaining contents of the bag (sorted)
# TODO: will probably need more inputs like current scores
# TODO: maybe better to pass around game state objects?
# TODO: these should probably also be objects
def random_strat(valid_plays, board, rack, bag):
    if valid_plays:
        return PLAY, random.choice(valid_plays)
    else:
        return PASS, None

def greedy_strat(valid_plays, board, rack, bag):
    if valid_plays:
        valid_plays.sort()
        valid_plays.reverse()
        return PLAY, valid_plays[0]
    else:
        return EXCHANGE, None

# TODO: write more strats (e.g., lookahead)

# ********** END STRATEGIES **********

def sim(strat1, strat2, log=False):
    board = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    bag = []
    for l, f in FREQS.items():
        for i in range(f):
            bag.append(l)
    random.shuffle(bag)
    racks = [[], []]
    draw(bag, racks[0])
    draw(bag, racks[1])
    scores = [0, 0]
    strats = [strat1, strat2]
    active = 0  # 0 on first player's turn, 1 on second player's turn
    first_play = True  # whether or not board is empty
    if log:
        print("\nGAME START!\n")
    wordless_turns = 0
    while racks[0] and racks[1]:
        # TODO: handle exchanges as possible moves
        valid_plays = find_valid_plays(board, racks[active], first_play)
        choice, details = strats[active](
            valid_plays, board, racks[active], sorted(bag))
        if choice == PLAY:
            score, r, c, p, edits, scored_words = details
            played_blank = False
            for rr, cc, l in edits:
                board[rr][cc] = l
            old_rack = racks[active][:]
            for l in p:
                if l.islower():
                    racks[active].remove('?')
                else:
                    racks[active].remove(l)
            scores[active] += score
            scoreline = "rack {}, played r{}c{} {}  score {}".format(
                rep_rack(old_rack), r+1, c+1, rep_words(scored_words), score)
            draw(bag, racks[active])
            wordless_turns = 0
            first_play = False
        elif choice == EXCHANGE and bag:
            wordless_turns += 1
            if bag:
                oldrack = racks[active][:]
                # currently just exchange everything. TODO: exchange specifics
                exchange(bag, racks[active], racks[active])
                scoreline = "rack {}, redrew to {}".format(
                    rep_rack(oldrack), rep_rack(racks[active]))
        else: # PASS
            wordless_turns += 1
            scoreline = "rack {}, passed".format(rep_rack(racks[active]))
        if log:
            print("{}-{}   P{} {}".format(
                scores[0], scores[1], active+1, scoreline))
            if choice == PLAY:
                print(rep_board(board))
            print("")
        if wordless_turns == 6:
            break
        active = 1 - active

    for p in range(2):
        if racks[p]:
            penalty = sum([VALUES[l] for l in racks[p]])
            scores[p] -= penalty
            if log:
                print("{}-{}   P{} left {} penalty -{}".format(
                    scores[0], scores[1], p+1, rep_rack(racks[p]), penalty))
    return scores

def compare_strats(strat1, strat2, num_trials):
    assert num_trials % 2 == 0, "Number of trials must be even"
    wins = [0, 0]
    score_totals = [0, 0]
    strats = [strat1, strat2]
    for i in range(num_trials):
        goes_first = i % 2
        scores = sim(strats[goes_first], strats[1-goes_first])
        if goes_first == 0:
            score1, score2 = scores
        else:
            score2, score1 = scores
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
    # TODO: print names of strats
    print("Player 1 won {}, Player 2 won {}".format(wins[0], wins[1]))
    print("Average scores: Player 1 {}, Player 2 {}".format(
        score_totals[0] / num_trials, score_totals[1] / num_trials))

compare_strats(random_strat, greedy_strat, 100)
