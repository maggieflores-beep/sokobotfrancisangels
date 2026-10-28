import heapq
from collections import deque

# u, d, l, r
DR = (-1, 1, 0, 0)
DC = (0, 0, -1, 1)
CH = ("u", "d", "l", "r")

# Weighted A*
WEIGHT = 5
TIME_LIMIT = 13
INF = 10 ** 9


class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        import time
        deadline = time.time() + TIME_LIMIT

        W, H = width, height
        N = W * H

        # read the board into flat arrays (cell index = r * W + c)
        wall = [True] * N          # out-of-range / '#' cells are walls
        goal = [False] * N
        boxes0 = []
        player0 = -1
        for r in range(H):
            for c in range(W):
                i = r * W + c
                wall[i] = mapData[r][c] == "#"
                goal[i] = mapData[r][c] == "."
                if itemsData[r][c] == "@":
                    player0 = i
                elif itemsData[r][c] == "$":
                    boxes0.append(i)
        goals = [i for i in range(N) if goal[i]]

        # neighbour table: nbr[d][i] = cell index after moving d from i, or -1
        nbr = [[-1] * N for _ in range(4)]
        for r in range(H):
            for c in range(W):
                i = r * W + c
                for d in range(4):
                    nr, nc = r + DR[d], c + DC[d]
                    if 0 <= nr < H and 0 <= nc < W and not wall[nr * W + nc]:
                        nbr[d][i] = nr * W + nc

        opp = (1, 0, 3, 2)

        # push distances: reverse BFS from every goal
        # box at b could have come from a = b - d if the player could stand at
        # a - d = b - 2d. Walls only; other boxes are ignored (lower bound).
        def pull_neighbours(b):
            for d in range(4):
                a = nbr[opp[d]][b]
                if a < 0:
                    continue
                p = nbr[opp[d]][a]
                if p < 0:
                    continue
                yield a

        push_dist = [INF] * N
        q = deque()
        for g in goals:
            push_dist[g] = 0
            q.append(g)
        while q:
            b = q.popleft()
            for a in pull_neighbours(b):
                if push_dist[a] == INF:
                    push_dist[a] = push_dist[b] + 1
                    q.append(a)
        dead = [(not wall[i]) and push_dist[i] == INF for i in range(N)]

        def heuristic(boxes):
            total = 0
            for b in boxes:
                total += push_dist[b]
            return total

        # freeze deadlock detection
        def frozen(pos, boxset, seen):
            if pos in seen:
                return True
            seen = seen | {pos}
            # horizontal axis
            left, right = nbr[2][pos], nbr[3][pos]
            h_blocked = (left < 0 or right < 0)
            if not h_blocked:
                if (left in boxset and frozen(left, boxset, seen)) or \
                   (right in boxset and frozen(right, boxset, seen)):
                    h_blocked = True
            if not h_blocked:
                if dead[left] and dead[right]:
                    h_blocked = True
            if not h_blocked:
                return False
            up, down = nbr[0][pos], nbr[1][pos]
            v_blocked = (up < 0 or down < 0)
            if not v_blocked:
                if (up in boxset and frozen(up, boxset, seen)) or \
                   (down in boxset and frozen(down, boxset, seen)):
                    v_blocked = True
            if not v_blocked:
                if dead[up] and dead[down]:
                    v_blocked = True
            return v_blocked

        def deadlocked(boxset, moved):
            # only the pushed box (and neighbours it freezes) can newly freeze
            if frozen(moved, boxset, frozenset()) and not goal[moved]:
                return True
            for d in range(4):
                n = nbr[d][moved]
                if n >= 0 and n in boxset and not goal[n] and frozen(n, boxset, frozenset()):
                    return True
            return False

        # start / trivial cases
        start_boxes = tuple(sorted(boxes0))
        if not goals or all(goal[b] for b in start_boxes):
            return ""

        # search
        # heap entry: (f, counter, g, player, boxes, parent_node)
        # parent_node is (previous_node, move_string)
        counter = 0
        h0 = heuristic(start_boxes)
        heap = [(WEIGHT * h0, 0, 0, player0, start_boxes, None)]
        best_g = {(player0, start_boxes): 0}
        closed = set()                       # (min reachable cell, boxes)

        while heap:
            if time.time() > deadline:
                return ""
            f, _, g, player, boxes, node = heapq.heappop(heap)
            boxset = set(boxes)

            # flood-fill once: every cell the player can reach without pushing
            came = {player: -1}
            came_dir = {}
            dq = deque([player])
            while dq:
                cur = dq.popleft()
                for d in range(4):
                    n = nbr[d][cur]
                    if n >= 0 and n not in came and n not in boxset:
                        came[n] = cur
                        came_dir[n] = d
                        dq.append(n)

            # same boxes + same reachable region = same state
            key = (min(came), boxes)
            if key in closed:
                continue
            closed.add(key)

            if all(goal[b] for b in boxes):
                parts = []
                while node is not None:
                    node, mv = node
                    parts.append(mv)
                return "".join(reversed(parts))

            for bi, b in enumerate(boxes):
                for d in range(4):
                    stand = nbr[opp[d]][b]       # player must be behind the box
                    if stand < 0 or stand not in came:
                        continue
                    to = nbr[d][b]               # where the box lands
                    if to < 0 or to in boxset or dead[to]:
                        continue

                    nb = list(boxes)
                    nb[bi] = to
                    nb.sort()
                    nb = tuple(nb)
                    if deadlocked(set(nb), to):
                        continue

                    # walk to stand, then push
                    walk = []
                    cur = stand
                    while cur != player:
                        walk.append(CH[came_dir[cur]])
                        cur = came[cur]
                    walk.reverse()
                    walk.append(CH[d])

                    ng = g + len(walk)
                    nstate = (b, nb)             # player ends on the old box cell
                    if ng < best_g.get(nstate, INF):
                        best_g[nstate] = ng
                        counter += 1
                        heapq.heappush(
                            heap,
                            (ng + WEIGHT * heuristic(nb), counter, ng, b, nb,
                             (node, "".join(walk))),
                        )

        return ""  # no solution