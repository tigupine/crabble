import itertools
import random

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
"""
BOARD = [[1,0,0,0,2,0,2,0,0,0,1],
         [0,1,0,0,0,2,0,0,0,1,0],
         [0,0,1,0,0,0,0,0,1,0,0],
         [0,0,0,3,0,0,0,3,0,0,0],
         [0,0,0,0,2,0,2,0,0,0,0],
         [0,2,0,0,0,1,0,0,0,2,0],
         [0,0,0,0,2,0,2,0,0,0,0],
         [0,0,0,3,0,0,0,3,0,0,0],
         [0,0,1,0,0,0,0,0,1,0,0],
         [0,1,0,0,0,2,0,0,0,1,0],
         [1,0,0,0,2,0,2,0,0,0,1]]
"""
BOARD_SIZE = len(BOARD)

# TODO: handle blanks? or not
VALUES = {'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4,
          'G': 2, 'H': 4, 'I': 1, 'J': 8, 'K': 5, 'L': 1,
          'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
          'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
          'Y': 4, 'Z': 10}

# Use about 1/3 the standard Scrabble frequencies since the board size is
# about 1/3 of the original. (Actually 40 tiles)
# TODO: replace one A with a blank?
FREQS = {'A': 4, 'B': 1, 'C': 1, 'D': 1, 'E': 4, 'F': 1,
         'G': 1, 'H': 1, 'I': 3, 'J': 1, 'K': 1, 'L': 1,
         'M': 1, 'N': 2, 'O': 3, 'P': 1, 'Q': 1, 'R': 2,
         'S': 2, 'T': 2, 'U': 1, 'V': 1, 'W': 1, 'X': 1,
         'Y': 1, 'Z': 1}

"""
# 1/2 the standard freqs
FREQS = {'A': 4, 'B': 1, 'C': 1, 'D': 2, 'E': 6, 'F': 1,
         'G': 1, 'H': 1, 'I': 4, 'J': 1, 'K': 1, 'L': 2,
         'M': 1, 'N': 3, 'O': 4, 'P': 1, 'Q': 1, 'R': 3,
         'S': 2, 'T': 3, 'U': 2, 'V': 1, 'W': 1, 'X': 1,
         'Y': 1, 'Z': 1}
"""

# Bonus for using entire rack
BINGO_BONUS = 15

f = open("words.txt", "r")
# TODO: expurgate
WORDS = set([l.strip() for l in f.readlines() if len(l) <= BOARD_SIZE + 1])

def undo_edits(board, edits):
    for rr, cc, _ in edits:
        board[rr][cc] = False

def check_board(board, edits, across, r, c, first_play):
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
    for i, across in checks:
        dr, dc = (0, 1) if across else (1, 0)
        w, score, multiplier, new_play = '', 0, 1, False
        rr, cc = i * dc, i * dr
        new_play = False
        while rr <= BOARD_SIZE and cc <= BOARD_SIZE:
            l = False
            if rr != BOARD_SIZE and cc != BOARD_SIZE:
                l = board[rr][cc]
            if l:
                w += l
                if l.islower():
                    new_play = True
                    m = BOARD[rr][cc]
                    if m == 1:
                        multiplier *= 2
                        score += VALUES[l.upper()]
                    else:
                        score += VALUES[l.upper()] * (m if m != 0 else 1)
                else:
                    score += VALUES[l]
            elif w:
                if len(w) >= 2:
                    # If the word has both uppercase and lowercase letters, it
                    # uses both new tiles and tiles that were on the board, so the
                    # board is connected.
                    if not w.isupper() and not w.islower():
                        connected = True
                    w = w.upper()
                    if w not in WORDS:
                        return False, None
                    if new_play:
                        total_score += score * multiplier
                        if len(edits) == RACK_SIZE:
                            total_score += BINGO_BONUS
                        scored_words.append(w)
                w, score, multiplier, new_play = '', 0, 1, False
            rr, cc = rr + dr, cc + dc
    if not connected and not first_play:
        return False, None
    return total_score, scored_words

def check_play(board, r, c, d, p, first_turn):
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
        board[rr][cc] = p[i].lower()
        edits.append((rr, cc, p[i]))
    if first_turn and not board[BOARD_SIZE//2][BOARD_SIZE//2]:
        undo_edits(board, edits)
        return False, None, None
    score, scored_words = check_board(board, edits, (dc == 1), r, c, first_turn)
    undo_edits(board, edits)
    return score, edits, scored_words
    
def find_plays(board, rack, first_play):
    plays = set()
    valid = []
    for n in range(1, RACK_SIZE + 1):
        for c in itertools.combinations(rack, n):
            for p in itertools.permutations(c):
                plays.add(p)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                continue
            for d in ((0, 1), (1, 0)):
                for p in plays:
                    score, edits, scored_words = check_play(
                        board, r, c, d, p, first_play)
                    if score:
                        valid.append(
                            (score, r, c, p, edits, scored_words))
    return valid

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
    return ', '.join(scored_words) 

# TODO: this should take two strategies as arguments
def sim(log=False):
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
    active = 0  # 0 on first player's turn, 1 on second player's turn
    first_play = True  # whether or not board is empty
    if log:
        print("\nGAME START!\n")
    wordless_turns = 0
    while racks[0] and racks[1]:
        # TODO: handle exchanges as possible moves
        plays = find_plays(board, racks[active], first_play)
        if plays:
            score, r, c, p, edits, scored_words = random.choice(plays)
            for rr, cc, l in edits:
                board[rr][cc] = l
            old_rack = racks[active][:]
            for l in p:
                racks[active].pop(racks[active].index(l))
            scores[active] += score
            scoreline = "rack {}, played r{}c{} {}  score {}".format(
                rep_rack(old_rack), r+1, c+1, rep_words(scored_words), score)
            draw(bag, racks[active])
            wordless_turns = 0
            first_play = False
        else:
            wordless_turns += 1
            if wordless_turns == 6:
                break
            if bag:
                oldrack = racks[active][:]
                # currently just exchange everything
                exchange(bag, racks[active], racks[active])
                scoreline = "rack {}, redrew to {}".format(
                    rep_rack(oldrack), rep_rack(racks[active]))
            else:
                scoreline = "rack {}, passed".format(rep_rack(racks[active]))
        if log:
            print("{}-{}   P{} {}".format(
                scores[0], scores[1], active+1, scoreline))
            if plays:
                print(rep_board(board))
            print("")
        active = 1 - active

    for p in range(2):
        if racks[p]:
            penalty = sum([VALUES[l] for l in racks[p]])
            scores[p] -= penalty
            if log:
                print("{}-{}   P{} left {} penalty -{}".format(
                    scores[0], scores[1], p+1, rep_rack(racks[p]), penalty))
    return scores

# sim(log=True)

wins = [0, 0]
for i in range(10000):
    print(i)
    score1, score2 = sim()
    print(score1, score2)
    if score1 > score2:
        wins[0] += 1
    else:
        wins[1] += 1

print(wins)
