from collections import namedtuple
import itertools
import random
from string import ascii_lowercase, ascii_uppercase

ValidPlay = namedtuple(
    'ValidPlay', ['score', 'edits', 'scored_words'])

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
def value(l):
    return VALUES.get(l, 0)

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

LEAVES = {}
f = open("adjusted_leaves.txt", "r")
values_total = 0
for l in f.readlines():
    leave, v = l.strip().split(',')
    vv = float(v)
    values_total += vv
    LEAVES[leave] = float(vv)
AVERAGE_LEAVE_VALUE = round(values_total / len(LEAVES), 1)

"""
# Fill in missing leave values.
# TODO move this elsewhere as its own function

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

# Estimate values
estimates = []
for n in range(1, RACK_SIZE):
    for r in itertools.product(ascii_uppercase, repeat=n):
        rack = ''.join(sorted(r))
        if not possible_rack(rack):
            continue
        if rack not in LEAVES:
            nb = nearby_racks(rack)
            if not nb:
                estimates.append((rack, AVERAGE_LEAVE_VALUE))
            else:
                estimates.append(
                    (rack, round(sum([LEAVES[rr] for rr in nb]) / len(nb), 1)))
for r, v in estimates:
    LEAVES[r] = v

f = open("adjusted_leaves.txt", "w")
for r, v in LEAVES.items():
    f.write("{},{}\n".format(r, v))
"""

# Options for strategies to return.
PLAY = 'PLAY'
EXCHANGE = 'EXCHANGE'
PASS = 'PASS'

# TODO: write tests for these functions

def check_board(board, edit_positions, across):
    checks = set()
    r, c = edit_positions[0]
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
                            board, edit_positions, across)
                        # Since we only have 1 blank, it is impossible to score 0.
                        if score:
                            valid_plays.append(
                                ValidPlay(
                                    score, tuple(edits), scored_words))
                        for rrr, ccc in edit_positions:
                            board[rrr][ccc] = False
                    
    return valid_plays

# Put the largest exchanges last so that strategies can use that.
def find_exchanges(rack):
    exch = []
    for n in range(1, RACK_SIZE + 1):
        seen = set()
        for c in itertools.combinations(rack, n):
            seen.add(tuple(sorted(c)))
        exch.extend([''.join(s) for s in seen])
    return exch

def draw(bag, rack):
    while bag and len(rack) < RACK_SIZE:
        rack.append(bag.pop())

def exchange(bag, rack, tiles):
    assert len(bag) >= len(tiles)
    exchanged = []
    for l in tiles:
        exchanged.append(rack.pop(rack.index(l)))
    draw(bag, rack)
    bag += exchanged
    random.shuffle(bag)

def rep_board(board):
    return '\n'.join([''.join([c if c else '.' for c in l]) for l in board])

def rep_rack(rack):
    return ''.join(sorted(rack))

def rep_words(scored_words):
    return ','.join(sorted(scored_words))

def copy(board):
    return [row[:] for row in board]

def remove_played_tiles(rack, edits):
    for _, _, l in edits:
        if l.islower():
            rack.remove('?')
        else:
            rack.remove(l)

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
def random_strat(valid_plays, valid_exchanges, board, rack, unseen,
                 tiles_in_bag):
    if valid_plays:
        return PLAY, random.choice(valid_plays)
    elif valid_exchanges:
        return EXCHANGE, random.choice(valid_exchanges)
    else:
        return PASS, None

# Choose a play with the highest score.
def greedy_strat(valid_plays, valid_exchanges, board, rack, unseen,
                 tiles_in_bag):
    if valid_plays:
        valid_plays.sort()
        valid_plays.reverse()
        return PLAY, valid_plays[0]
    elif valid_exchanges:
        return EXCHANGE, valid_exchanges[-1] # one of the biggest possible
    else:
        return PASS, None
    
    return PLAY, valid_plays[0]

def leave_strat_m(m):
    return (lambda valid_plays, valid_exchanges, board, rack, unseen,
            tiles_in_bag : leave_strat(
                valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, m=m))

# Adjust the score of each potential move based on data from two future moves.
def leave_strat(valid_plays, valid_exchanges, board, rack, unseen,
                tiles_in_bag, m=0.2):
    best_adj_score = -1000
    best_choice = PASS, None
    for v in valid_plays:
        r_copy = rack[:]
        remove_played_tiles(r_copy, v.edits)
        r_copy.sort()
        adj_score = v.score + m * LEAVES[''.join(r_copy)]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = PLAY, v
    for ex in find_exchanges(rack):
        if len(ex) > tiles_in_bag:
            continue
        if len(ex) == RACK_SIZE:
            adj_score = m * AVERAGE_LEAVE_VALUE
        else:
            adj_score = m * LEAVES[ex]
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_choice = EXCHANGE, ex
    return best_choice

# Assume greedy opponent, see what they do next turn.
# TODO: aggh this is slow, mostly because of the blank
def lookahead_1_strat(valid_plays, valid_exchanges, board, rack, unseen,
                      tiles_in_bag):
    if not valid_plays:
        if valid_exchanges:
            return EXCHANGE, valid_exchanges[-1]
        else:
            return PASS, None
    best_delta = -1000
    best_play = None
    valid_plays.sort()
    valid_plays.reverse()
    # Only look at the highest-scoring valid plays, to save time.
    for v in valid_plays[0 : min(len(valid_plays), 10)]:
        # TODO: this is code from sim(). Factor out to remove redundancy.
        score, edits, ___ = v
        for rr, cc, l in edits:
            board[rr][cc] = l
        avg_opp_score = 0
        num_trials = 10
        for _ in range(num_trials):
            # Simulate a rack for the opponent.
            opp_rack = random.sample(
                unseen, max(0, min(len(unseen)-len(edits), RACK_SIZE)))
            opp_valid_plays = find_valid_plays(board, opp_rack, False)
            # Skip the last three inputs because greedy_strat doesn't use them.
            choice, details = greedy_strat(opp_valid_plays, None, None, None)
            if choice == PLAY:
                avg_opp_score += details[0]
        for rr, cc, _ in edits:
            board[rr][cc] = False
        avg_opp_score /= num_trials
        delta = score - avg_opp_score
        if delta > best_delta:
            best_delta = delta
            best_play = v
    return PLAY, best_play

def endgame_strat(valid_plays, board, rack, unseen, tiles_in_bag):
    if len(unseen) <= 10:
        return lookahead_1_strat(valid_plays, valid_exchanges, board, rack,
                                 unseen, tiles_in_bag)
    else:
        return greedy_strat(valid_plays, valid_exchanges, board, rack, unseen,
                            tiles_in_bag)
    
# TODO: write more strats (e.g., lookahead 2?). Try to use CS238 material!

# ********** END STRATEGIES **********

def sim(strat1, strat2, log=False):
    strats = [strat1, strat2]
    scores = [0, 0]  
    record = [[], []]
    board = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    bag = []
    for l, f in FREQS.items():
        for i in range(f):
            bag.append(l)
    random.shuffle(bag)
    racks = [[], []]
    draw(bag, racks[0])
    draw(bag, racks[1])
    active = 0  # 0 on first player's turn, 1 on second player's turn
    first_play = True  # whether or not board is empty
    if log:
        print("\nGAME START!\n")
    wordless_turns = 0 # the game ends automatically after 6 wordless turns
    while racks[0] and racks[1]: # or when a rack is empty
        valid_plays = find_valid_plays(board, racks[active], first_play)
        valid_exchanges = (
            find_exchanges(racks[active]) if len(bag) >= RACK_SIZE else [])
        choice, details = strats[active](
            valid_plays, valid_exchanges, board, racks[active],
                sorted(racks[1-active] + bag), len(bag))
        if choice == PLAY:
            score, edits, scored_words = details
            for rr, cc, l in edits:
                board[rr][cc] = l
            old_rack = racks[active][:]
            remove_played_tiles(racks[active], edits)
            scores[active] += score
            scoreline = "rack {}, played r{}c{} {}  score {}".format(
                rep_rack(old_rack), edits[0][0]+1, edits[0][1]+1,
                rep_words(scored_words), score)
            record[active].append((PLAY, score, rep_rack(racks[active])))
            draw(bag, racks[active])
            wordless_turns = 0
            first_play = False
        elif choice == EXCHANGE:
            wordless_turns += 1
            if bag:
                oldrack = racks[active][:]
                exchange(bag, racks[active], details)
                scoreline = "rack {}, exchanged {}, redrew to {}".format(
                    rep_rack(oldrack), details, rep_rack(racks[active]))
                record[active].append(
                    (EXCHANGE, 0, rep_rack(racks[active])))
        else: # PASS
            wordless_turns += 1
            scoreline = "rack {}, passed".format(rep_rack(racks[active]))
            record[active].append((PASS, 0, rep_rack(racks[active])))
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
    return scores, record

def compare_strats(strat1, strat2, num_trials, log_each_game=False,
                   progress_update_every = 100):
    assert num_trials % 2 == 0, "Number of trials must be even"
    wins = [0, 0]
    score_totals = [0, 0]
    strats = [strat1, strat2]
    for i in range(1, num_trials+1):
        goes_first = i % 2
        scores, _ = sim(strats[goes_first], strats[1-goes_first], log=log_each_game)
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
    # TODO: print names of strats
    print("Player 1 won {}, Player 2 won {}".format(wins[0], wins[1]))
    print("Average scores: Player 1 {}, Player 2 {}".format(
        score_totals[0] / num_trials, score_totals[1] / num_trials))
    return wins

def compare_strats_with_confidence(
    strat1, strat2, num_experiments, num_trials_each):
    s1s = []
    for i in range(num_experiments):
        print("Experiment {} of {}".format(i + 1, num_experiments))
        s1, _ = compare_strats(
            leave_strat, greedy_strat, num_trials_each, log_each_game=False,
            progress_update_every=100)
        s1s.append(s1)

    s1mean = sum(s1s) / num_experiments
    s1stdev = (sum([(x - s1mean)**2 for x in s1s]) / (num_experiments - 1))**0.5
    print("Strat 1 mean wins: {} stdev: {}".format(
        round(s1mean, 2), round(s1stdev, 2)))

def compile_leave_data(num_trials, min_instances=10, log_every=100):
    per_leave_data = {}
    for t in range(num_trials):
        if t % log_every == 0:
            print(t)
        _, record = sim(greedy_strat, greedy_strat, log=False)
        for i in range(2):
            for j in range(len(record[i])-2):
                _, _, rack = record[i][j]
                if len(rack) == RACK_SIZE:
                    continue
                move_1, score_1, _ = record[i][j+1]
                move_2, score_2, _ = record[i][j+2]
                # Don't bias too heavily toward the endgame.
                if move_1 != PLAY and move_2 != PLAY:
                    continue
                score = score_1 + score_2
                if rack not in per_leave_data:
                    per_leave_data[rack] = [0, 0]
                per_leave_data[rack][0] += score
                per_leave_data[rack][1] += 1
    f = open("leaves.txt", "w")
    for k, v in sorted(per_leave_data.items()):
        total, instances = v
        if instances >= min_instances:
            f.write("{},{}\n".format(k, round(v[0]/v[1], 1)))

#compile_leave_data(100000)
#sim(leave_strat, leave_strat, log=True)
#compare_strats_with_confidence(leave_strat, greedy_strat, 20, 1000)
for i in range(1, 20):
    print(i*0.1)
    compare_strats(leave_strat_m(i*0.1), greedy_strat, 1000,
                   progress_update_every=10000)
