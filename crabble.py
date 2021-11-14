from collections import namedtuple
import itertools
import random
from string import ascii_lowercase

ValidPlay = namedtuple(
    'ValidPlay', ['score', 'r', 'c', 'edits', 'scored_words'])

RACK_SIZE = 5
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
          'Y': 4, 'Z': 10, '?': 0}

# 40 tiles
# either do 3 As and one ?, or 4 As
FREQS = {'A': 4, 'B': 1, 'C': 1, 'D': 1, 'E': 4, 'F': 1,
         'G': 1, 'H': 1, 'I': 3, 'J': 1, 'K': 1, 'L': 1,
         'M': 1, 'N': 2, 'O': 3, 'P': 1, 'Q': 1, 'R': 2,
         'S': 2, 'T': 2, 'U': 1, 'V': 1, 'W': 1, 'X': 1,
         'Y': 1, 'Z': 1, '?': 0}

# Bonus for using entire rack
BINGO_BONUS = 20

f = open("words.txt", "r")
WORDS = set([l.strip() for l in f.readlines() if len(l) <= BOARD_SIZE + 1])

# Options for strategies to return.
PLAY = 'PLAY'
EXCHANGE = 'EXCHANGE'
PASS = 'PASS'

# TODO: write tests for these functions

def check_board(board, edit_positions, across, r, c):
    def value(l):
        return VALUES.get(l, 0)
    checks = set()
    if across:
        checks.add((r, True))
        for _, c in edit_positions:
            checks.add((c, False))
    else:
        checks.add((c, False))
        for r, _ in edit_positions:
            checks.add((r, True))
    total_score = 0
    scored_words = []
    for i, across in checks:
        dr, dc = (0, 1) if across else (1, 0)
        w, score, multiplier, has_new = '', 0, 1, False
        rr, cc = i * dc, i * dr
        while rr <= BOARD_SIZE and cc <= BOARD_SIZE:
            l = False
            if rr != BOARD_SIZE and cc != BOARD_SIZE:
                l = board[rr][cc]
            if l:
                w += l
                if (rr, cc) in edit_positions:
                    has_new = True
                    m = PREMIUMS[rr][cc]
                    if m == 1:
                        multiplier *= 2
                        score += value(l)
                    else:
                        score += value(l) * (m if m != 0 else 1)
                else:
                    score += value(l)
            elif w:  # reached the end of a word, process it
                if len(w) >= 2:
                    if w.upper() not in WORDS:
                        return False, None
                    if has_new:
                        total_score += score * multiplier
                        scored_words.append(w)
                w, score, multiplier, has_new = '', 0, 1, False
            rr, cc = rr + dr, cc + dc
    if len(edit_positions) == RACK_SIZE:
        total_score += BINGO_BONUS
    return total_score, tuple(scored_words)

def check_lane(board, neighbors, r, c, d, max_tiles):
    placement_info = []
    connected = False
    rr, cc = r, c
    dr, dc = d
    results = []
    # See how many tiles need to be played for connectedness.
    for i in range(max_tiles):
        while rr < BOARD_SIZE and cc < BOARD_SIZE and board[rr][cc]:
            rr, cc = rr + dr, cc + dc
        if rr == BOARD_SIZE or cc == BOARD_SIZE:
            return placement_info
        if neighbors[rr][cc]:
            connected = True
        placement_info.append((rr, cc, connected))
        # pretend to place a tile
        rr, cc = rr + dr, cc + dc
    return placement_info

def neighboring_cells(board):
    neighbors = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                        and not board[rr][cc]):
                        neighbors[rr][cc] = True
    return neighbors

def find_valid_plays(board, rack, first_play):
    valid_plays = []
    rack_len = len(rack)
    tile_sets = [set() for _ in range(rack_len)]
    for num_tiles in range(1, rack_len + 1):
        for com in itertools.combinations(rack, num_tiles):
            for perm in itertools.permutations(com):
                if '?' in perm:
                    blank_index = perm.index('?')
                    for l in ascii_lowercase:
                        tile_sets[num_tiles-1].add(
                            perm[0:blank_index] + (l,) + perm[blank_index+1:])
                else:
                    tile_sets[num_tiles-1].add(perm)
    neighbors = neighboring_cells(board)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]:
                continue
            # small optimization, could probably remove
            if first_play and not (r == BOARD_SIZE // 2 or c == BOARD_SIZE // 2):
                continue
            for d in ((0, 1), (1, 0)):
                placement_info = check_lane(board, neighbors, r, c, d, rack_len)
                if first_play:
                    used_center_square = False
                    for rr, cc, _ in placement_info:
                        if rr == BOARD_SIZE // 2 and cc == BOARD_SIZE // 2:
                            used_center_square = True
                    if not used_center_square:
                        continue
                edit_positions = []
                across = d[0] == 0
                for i in range(len(placement_info)):
                    rr, cc, connected = placement_info[i]
                    edit_positions.append((rr, cc))
                    if (not first_play) and (not connected):
                        continue
                    for ts in tile_sets[i]:
                        edits = []
                        for j in range(i+1):
                            rrr, ccc = edit_positions[j]
                            board[rrr][ccc] = ts[j]
                            edits.append((rrr, ccc, ts[j]))
                        score, scored_words = check_board(
                            board, edit_positions, across, r, c)
                        # Since we only have 1 blank, it is impossible to score 0.
                        if score:
                            valid_plays.append(
                                ValidPlay(
                                    score, r, c, tuple(edits), scored_words))
                        for rrr, ccc in edit_positions:
                            board[rrr][ccc] = False
                    
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

def copy(board):
    return [row[:] for row in board]

# ********** BEGIN STRATEGIES **********

# A strategy takes the following inputs:
# valid_plays: list of all valid plays, each as a ValidPlay tuple
# board: the current board
# rack: the active player's rack
# unseen: the remaining contents of the bag + opponent's rack (sorted)
# TODO: will probably need more inputs like current scores
# TODO: maybe better to pass around game state objects?
# TODO: these strategies themselves should probably also be objects

# Choose a valid play uniformly at random.
def random_strat(valid_plays, board, rack, unseen):
    if valid_plays:
        return PLAY, random.choice(valid_plays)
    else:
        return EXCHANGE, None

# Choose a play with the highest score.
def greedy_strat(valid_plays, board, rack, unseen):
    if not valid_plays:
        return EXCHANGE, None
    valid_plays.sort()
    valid_plays.reverse()
    return PLAY, valid_plays[0]

# Choose a play with the highest score, adjusted to subtract the value of the
# tiles left in the rack.
def leave_strat(valid_plays, board, rack, unseen, m=1):
    if not valid_plays:
        return EXCHANGE, None
    best_delta = -1000
    best_play = None
    valid_plays.sort()
    valid_plays.reverse()
    for v in valid_plays:
        leave = rack[:]
        for _, _, l in v.edits:
            if l.islower():
                leave.remove('?')
            else:
                leave.remove(l)
        # Assume high-valued letters are harder to play.
        # TODO: try to learn this with a function?
        penalty = 0
        for l in leave:
            penalty += VALUES[l]
        delta = v.score - m * penalty
        if delta > best_delta:
            best_delta = delta
            best_play = v
    return PLAY, best_play  

# Assume greedy opponent, see what they do next turn.
# TODO: aggh this is slow, mostly because of the blank
def lookahead_1_strat(valid_plays, board, rack, unseen):
    if not valid_plays:
        return EXCHANGE, None
    best_delta = -1000
    best_play = None
    valid_plays.sort()
    valid_plays.reverse()
    # Only look at the highest-scoring valid plays, to save time.
    for v in valid_plays[0 : min(len(valid_plays), 10)]:
        # TODO: this is code from sim(). Factor out to remove redundancy.
        new_board = copy(board)
        score, _, __, edits, ___ = v
        for rr, cc, l in edits:
            new_board[rr][cc] = l
        avg_opp_score = 0
        num_trials = 10
        for _ in range(num_trials):
            # Simulate a rack for the opponent.
            opp_rack = random.sample(
                unseen, max(0, min(len(unseen)-len(edits), RACK_SIZE)))
            opp_valid_plays = find_valid_plays(new_board, opp_rack, False)
            # Skip the last three inputs because greedy_strat doesn't use them.
            choice, details = greedy_strat(opp_valid_plays, None, None, None)
            if choice == PLAY:
                avg_opp_score += details[0]
        avg_opp_score /= num_trials
        delta = score - avg_opp_score
        if delta > best_delta:
            best_delta = delta
            best_play = v
    return PLAY, best_play 
    
# TODO: write more strats (e.g., lookahead 2?). Try to use CS238 material!

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
            valid_plays, board, racks[active], sorted(racks[1-active] + bag))
        if choice == PLAY:
            score, r, c, edits, scored_words = details
            for rr, cc, l in edits:
                board[rr][cc] = l
            old_rack = racks[active][:]
            for _, _, l in edits:
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

def compare_strats(strat1, strat2, num_trials, log_each_game=False):
    assert num_trials % 2 == 0, "Number of trials must be even"
    wins = [0, 0]
    score_totals = [0, 0]
    strats = [strat1, strat2]
    for i in range(num_trials):
        goes_first = i % 2
        scores = sim(strats[goes_first], strats[1-goes_first], log=log_each_game)
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

compare_strats(greedy_strat, leave_strat, 1000, log_each_game=False)
