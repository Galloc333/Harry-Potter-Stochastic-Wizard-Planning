
ids = ["318931672", "208302661"]

import collections
from collections import deque
import itertools
import copy
import time

GLOBAL_SCORE = 0

RS_ACT = "reset"
TERM_ACT = "terminate"
MOV_ACT = "move"
DEST_ACT = "destroy"
WAIT_ACT = "wait"

DEST_R = 2
RESET_R = -2
DE_CATCH_R = -1

class FatherAgent:
    def __init__(self, conf):
        self.cnf = copy.deepcopy(conf)
        self.t_left = self.cnf["turns_to_go"]
        self.w_keys = sorted(self.cnf["wizards"].keys())
        self.d_keys = sorted(self.cnf["death_eaters"].keys())
        self.h_keys = sorted(self.cnf["horcrux"].keys())
        self.free_cells, self.neigh_map = self._build_graph(self.cnf["map"])
        self.d_paths = {k: self.cnf["death_eaters"][k]["path"] for k in self.d_keys}
        self.h_poss = {}
        for h in self.h_keys:
            orig = self.cnf["horcrux"][h]["location"]
            poss = set(self.cnf["horcrux"][h]["possible_locations"]) | {orig}
            self.h_poss[h] = list(poss)

    def _build_graph(self, grid):
        rnum = len(grid)
        cnum = len(grid[0]) if rnum > 0 else 0
        free = {(r, c) for r in range(rnum) for c in range(cnum) if grid[r][c] != 'I'}
        nb_map = {}
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for cell in free:
            nbrs = []
            for dr, dc in steps:
                cand = (cell[0] + dr, cell[1] + dc)
                if cand in free:
                    nbrs.append(cand)
            nb_map[cell] = nbrs
        return free, nb_map

    def _serialize(self, st):
        w_tuple = tuple(st["wizards"][w]["location"] for w in self.w_keys)
        d_tuple = tuple(st["death_eaters"][d]["index"] for d in self.d_keys)
        h_tuple = tuple(st["horcrux"][h]["location"] for h in self.h_keys)
        return (w_tuple, d_tuple, h_tuple, st["turns_to_go"])

    def _compute_actions(self, s_ser, t0=None, t_lim=None, chk=False):
        w_pos, d_idx, h_pos, t_rem = s_ser
        if t_rem <= 0:
            return [TERM_ACT]
        acts = [RS_ACT, TERM_ACT]
        all_moves = []
        for i, w in enumerate(self.w_keys):
            cur = w_pos[i]
            moves = []
            if cur in self.neigh_map:
                for nb in self.neigh_map[cur]:
                    moves.append((MOV_ACT, w, nb))
            moves.append((WAIT_ACT, w))
            for j, h in enumerate(self.h_keys):
                if h_pos[j] == cur:
                    moves.append((DEST_ACT, w, h))
            all_moves.append(moves)
        for joint in itertools.product(*all_moves):
            if chk and (time.perf_counter() - t0 >= t_lim):
                self.finish_BFS = False
                self.finish_VI = False
                return None
            acts.append(joint)
        return acts

    def _compute_transition_probs(self, s_ser, act, t0=None, t_lim=None, chk=False):
        w_pos, d_idx, h_pos, t_rem = s_ser
        if t_rem <= 0:
            return {s_ser: 1.0}
        if act == RS_ACT:
            init_ser = self._serialize(self.cnf)
            w0, d0, h0, _ = init_ser
            return {(w0, d0, h0, t_rem - 1): 1.0}
        if act == TERM_ACT:
            return {(w_pos, d_idx, h_pos, 0): 1.0}
        new_w = list(w_pos)
        if isinstance(act, tuple):
            w_map = {w: i for i, w in enumerate(self.w_keys)}
            for single in act:
                if chk and (time.perf_counter() - t0 >= t_lim):
                    return None
                if single[0] == MOV_ACT:
                    new_w[w_map[single[1]]] = single[2]
        new_w = tuple(new_w)
        def d_options(did, cur):
            path_seq = self.d_paths[did]
            L = len(path_seq)
            if L == 1:
                return [(cur, 1.0)]
            if cur == 0:
                return [(0, 0.5), (1, 0.5)]
            if cur == L - 1:
                return [(L - 1, 0.5), (L - 2, 0.5)]
            return [(cur - 1, 1/3), (cur, 1/3), (cur + 1, 1/3)]
        d_opts_list = []
        for pos, did in zip(d_idx, self.d_keys):
            if chk and (time.perf_counter() - t0 >= t_lim):
                return None
            d_opts_list.append(d_options(did, pos))
        def h_options(hid, cur):
            p_ch = self.cnf["horcrux"][hid]["prob_change_location"]
            poss = self.h_poss[hid]
            stay = 1 - p_ch
            mv = p_ch / len(poss)
            opts = []
            for pos in poss:
                prob = mv + (stay if pos == cur else 0)
                if prob > 0:
                    opts.append((pos, prob))
            return opts
        h_opts_list = []
        for i, hid in enumerate(self.h_keys):
            h_opts_list.append(h_options(hid, h_pos[i]))
        next_probs = {}
        for d_choice in itertools.product(*d_opts_list):
            if chk and (time.perf_counter() - t0 >= t_lim):
                self.finish_BFS = False
                self.finish_VI = False
                return None
            p_d = 1.0
            new_d = [None] * len(self.d_keys)
            for i, (n_idx, p_val) in enumerate(d_choice):
                p_d *= p_val
                new_d[i] = n_idx
            if p_d <= 0:
                continue
            new_d = tuple(new_d)
            for h_choice in itertools.product(*h_opts_list):
                if chk and (time.perf_counter() - t0 >= t_lim):
                    self.finish_BFS = False
                    self.finish_VI = False
                    return None
                p_h = 1.0
                new_h = [None] * len(self.h_keys)
                for i, (n_pos, p_val) in enumerate(h_choice):
                    p_h *= p_val
                    new_h[i] = n_pos
                if p_h <= 0:
                    continue
                new_h = tuple(new_h)
                tot = p_d * p_h
                ns = (new_w, new_d, new_h, t_rem - 1)
                next_probs[ns] = next_probs.get(ns, 0) + tot
        return next_probs

    def _compute_reward(self, s_ser, act, ns_ser):
        rew = 0
        w_pos, d_idx, h_pos, _ = s_ser
        new_w, new_d, new_h, _ = ns_ser
        if act == RS_ACT:
            rew += RESET_R
        if isinstance(act, tuple):
            for a in act:
                if a[0] == DEST_ACT:
                    rew += DEST_R
        for i, pos in enumerate(new_w):
            for j, did in enumerate(self.d_keys):
                seq = self.d_paths[did]
                if seq[new_d[j]] == pos:
                    rew += DE_CATCH_R
        return rew

    def _nearest_viable_target(self, start, targets, avoid):
        if start in targets:
            return start
        seen = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nb in self.neigh_map.get(cur, []):
                if nb in avoid or nb in seen:
                    continue
                if nb in targets:
                    return nb
                seen.add(nb)
                queue.append(nb)
        return None

    def _bfs_next_step(self, start, goal, avoid):
        if start == goal:
            return start
        if start not in self.free_cells or goal not in self.free_cells:
            return None
        parent = {}
        visited = {start}
        q = deque([start])
        found = False
        while q and not found:
            cur = q.popleft()
            for nb in self.neigh_map.get(cur, []):
                if nb in avoid or nb in visited:
                    continue
                visited.add(nb)
                parent[nb] = cur
                if nb == goal:
                    found = True
                    break
                q.append(nb)
        if not found:
            return None
        path = [goal]
        while path[-1] in parent:
            path.append(parent[path[-1]])
            if path[-1] == start:
                break
        path.reverse()
        return path[1] if len(path) > 1 else start

class WizardAgent(FatherAgent):
    def __init__(self, conf):
        super().__init__(conf)
        self.finish_BFS = True
        self.finish_VI = True
        self.total_runtime = 240
        self.bfs_runtime = 80
        self.V = [dict() for _ in range(self.t_left + 1)]
        self.policy = [dict() for _ in range(self.t_left + 1)]
        self.prev_conf = None
        self.prev_act = None
        self._partial_build_and_vi()

    def act(self, st):
        global GLOBAL_SCORE
        if self.prev_conf is not None and self.prev_act is not None:
            self._update_score(self.prev_conf, self.prev_act, st)
        turns = st["turns_to_go"]
        if turns <= 0:
            self.prev_conf = st
            self.prev_act = TERM_ACT
            return TERM_ACT
        ser = self._serialize(st)
        if self.finish_BFS and self.finish_VI and turns < len(self.policy) and ser in self.policy[turns]:
            chosen = self.policy[turns][ser]
            self.prev_conf = st
            self.prev_act = chosen
            return chosen
        fb = self._heuristic_act(st)
        self.prev_conf = st
        self.prev_act = fb
        return fb

    def _update_score(self, prev, act, new):
        global GLOBAL_SCORE
        if act == RS_ACT:
            GLOBAL_SCORE += RESET_R
        elif act == TERM_ACT:
            pass
        elif isinstance(act, tuple):
            for a in act:
                if a[0] == DEST_ACT:
                    GLOBAL_SCORE += DEST_R
        for w, dat in new["wizards"].items():
            loc = dat["location"]
            for d, ddat in new["death_eaters"].items():
                idx = ddat["index"]
                if ddat["path"][idx] == loc:
                    GLOBAL_SCORE += DE_CATCH_R

    def _heuristic_act(self, st):
        if st["turns_to_go"] <= 0:
            return TERM_ACT
        danger = set()
        for d in self.d_keys:
            idx = st["death_eaters"][d]["index"]
            danger.add(self.d_paths[d][idx])
        joint = []
        for w in self.w_keys:
            cur = st["wizards"][w]["location"]
            found = None
            for h in self.h_keys:
                if cur == st["horcrux"][h]["location"]:
                    found = (DEST_ACT, w, h)
                    break
            if found:
                joint.append(found)
            else:
                hset = {st["horcrux"][h]["location"] for h in self.h_keys}
                tgt = self._nearest_viable_target(cur, hset, danger)
                if tgt is None:
                    joint.append((WAIT_ACT, w))
                else:
                    nxt = self._bfs_next_step(cur, tgt, danger)
                    if nxt is None or nxt == cur:
                        joint.append((WAIT_ACT, w))
                    else:
                        joint.append((MOV_ACT, w, nxt))
        return tuple(joint)

    def _partial_build_and_vi(self):
        t0 = time.perf_counter()
        states, _ = self._partial_bfs(self.bfs_runtime, t0)
        used = time.perf_counter() - t0
        rem = self.total_runtime - used
        if rem <= 0:
            self.finish_VI = False
            return
        self._run_value_iteration(states, rem)

    def _partial_bfs(self, limit, t0):
        term_count = 0
        init_ser = self._serialize(self.cnf)
        if init_ser[-1] == 0:
            self.V[0][init_ser] = 0.0
            term_count += 1
        q = deque([init_ser])
        seen = {init_ser}
        while q:
            if time.perf_counter() - t0 >= limit:
                self.finish_BFS = False
                return seen, term_count
            curr = q.popleft()
            curr_turn = curr[-1]
            if curr_turn <= 0:
                continue
            acts = self._compute_actions(curr, t0, limit, True)
            if acts is None:
                return seen, term_count
            for a in acts:
                if time.perf_counter() - t0 >= limit:
                    self.finish_BFS = False
                    return seen, term_count
                trans = self._compute_transition_probs(curr, a, t0, limit, True)
                if trans is None:
                    return seen, term_count
                for ns, p in trans.items():
                    if time.perf_counter() - t0 >= limit:
                        self.finish_BFS = False
                        return seen, term_count
                    if p > 0 and ns not in seen:
                        if ns[-1] == 0:
                            self.V[0][ns] = 0.0
                            term_count += 1
                        seen.add(ns)
                        q.append(ns)
        return seen, term_count

    def _run_value_iteration(self, states, vi_lim):
        t0 = time.perf_counter()
        for t in range(1, self.t_left + 1):
            states_t = [s for s in states if s[-1] == t]
            for s in states_t:
                if time.perf_counter() - t0 >= vi_lim:
                    self.finish_VI = False
                    return
                acts = self._compute_actions(s, t0, vi_lim, True)
                if acts is None:
                    return
                if not acts:
                    self.V[t][s] = 0.0
                    self.policy[t][s] = TERM_ACT
                    continue
                best = float('-inf')
                best_a = None
                for a in acts:
                    if time.perf_counter() - t0 >= vi_lim:
                        self.finish_VI = False
                        return
                    trans = self._compute_transition_probs(s, a, t0, vi_lim, True)
                    if trans is None:
                        return
                    q_val = 0.0
                    for ns, p in trans.items():
                        if time.perf_counter() - t0 >= vi_lim:
                            self.finish_VI = False
                            return
                        r = self._compute_reward(s, a, ns)
                        q_val += p * (r + self.V[ns[-1]].get(ns, 0.0))
                    if q_val > best:
                        best = q_val
                        best_a = a
                self.V[t][s] = best
                self.policy[t][s] = best_a

class OptimalWizardAgent(FatherAgent):
    def __init__(self, conf):
        super().__init__(conf)
        self.V = [dict() for _ in range(self.t_left + 1)]
        self.policy = [dict() for _ in range(self.t_left + 1)]
        self.all_states = self._build_reachable_states()
        for s in self.all_states:
            if s[-1] == 0:
                self.V[0][s] = 0.0
        for t in range(1, self.t_left + 1):
            sts = [s for s in self.all_states if s[-1] == t]
            for s in sts:
                acts = self._compute_actions(s)
                if not acts:
                    self.V[t][s] = 0.0
                    self.policy[t][s] = TERM_ACT
                    continue
                best_val = float('-inf')
                best_act = None
                for a in acts:
                    trans = self._compute_transition_probs(s, a)
                    q_val = 0.0
                    for ns, p in trans.items():
                        r_val = self._compute_reward(s, a, ns)
                        q_val += p * (r_val + self.V[ns[-1]].get(ns, 0.0))
                    if q_val > best_val:
                        best_val = q_val
                        best_act = a
                self.V[t][s] = best_val
                self.policy[t][s] = best_act

    def _build_reachable_states(self):
        init = self._serialize(self.cnf)
        q = deque([init])
        reached = {init}
        while q:
            curr = q.popleft()
            if curr[-1] <= 0:
                continue
            for a in self._compute_actions(curr):
                trans = self._compute_transition_probs(curr, a)
                for ns, p in trans.items():
                    if p > 0 and ns not in reached:
                        reached.add(ns)
                        q.append(ns)
        return reached

    def act(self, st):
        rem = st["turns_to_go"]
        if rem <= 0:
            return TERM_ACT
        ser = self._serialize(st)
        if ser not in self.policy[rem]:
            return tuple((WAIT_ACT, w) for w in self.w_keys)
        return self.policy[rem][ser]
