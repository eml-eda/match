import random
import numpy as np

from match.opt.passes.schedule_pass import MatchScheduleOptPass
from match.schedule.schedule import MatchSchedule


class MatchGATilingPass(MatchScheduleOptPass):
    """GA-based tiling to fit low-level memories.

    Variables: per-loop tile sizes (divisors). Fitness penalizes overflow.
    Applies updated sizes to schedule.tensor_tiles.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = {
            "population_size": 24,
            "generations": 40,
            "crossover_rate": 0.9,
            "mutation_rate": 0.25,
            "elitism": 2,
            "seed": 0xC0FFEE,
        }
        if config:
            self.config.update(config)
        self.rng = random.Random(self.config["seed"])

    def __call__(self, schedule: MatchSchedule) -> MatchSchedule:
        if not schedule.tensor_tiles:
            schedule.set_default_tensor_tiles()
        cap_bytes = self._get_lowest_mem_capacity(schedule)
        if cap_bytes is None:
            return schedule
        # Build GA inputs
        loop_extents = [[max(1, lp.size) for lp in blk.loops] for blk in schedule.blocks]
        choices = [[self._divisors(n) for n in blke] for blke in loop_extents]
        pop = [self._random_individual(choices) for _ in range(self.config["population_size"])]
        fits = [self._fitness(ind, schedule, cap_bytes) for ind in pop]
        for _ in range(self.config["generations"]):
            new_pop = []
            elite_idx = sorted(range(len(pop)), key=lambda i: fits[i])[: self.config["elitism"]]
            new_pop.extend([pop[i] for i in elite_idx])
            while len(new_pop) < self.config["population_size"]:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2, choices)
                c1 = self._mutate(c1, choices)
                c2 = self._mutate(c2, choices)
                new_pop.append(c1)
                if len(new_pop) < self.config["population_size"]:
                    new_pop.append(c2)
            pop = new_pop
            fits = [self._fitness(ind, schedule, cap_bytes) for ind in pop]
        best = pop[min(range(len(pop)), key=lambda i: fits[i])]
        self._apply(best, schedule)
        return schedule

    # --- helpers ---
    def _divisors(self, n: int):
        return [d for d in range(1, n + 1) if n % d == 0]

    def _random_individual(self, choices):
        return [[self.rng.choice(opts) if opts else 1 for opts in blk] for blk in choices]

    def _tournament(self, pop, fits):
        k = 3
        cand = [self.rng.randrange(len(pop)) for _ in range(k)]
        return pop[min(cand, key=lambda i: fits[i])]

    def _crossover(self, a, b, choices):
        if self.rng.random() > self.config["crossover_rate"]:
            return a, b
        c1, c2 = [], []
        for ba, bb in zip(a, b):
            n = len(ba)
            if n == 0:
                c1.append([])
                c2.append([])
                continue
            cut = self.rng.randrange(n)
            c1.append(ba[:cut] + bb[cut:])
            c2.append(bb[:cut] + ba[cut:])
        return c1, c2

    def _mutate(self, g, choices):
        if self.rng.random() < self.config["mutation_rate"]:
            bi = self.rng.randrange(len(g)) if g else 0
            if g and g[bi]:
                li = self.rng.randrange(len(g[bi]))
                opts = choices[bi][li]
                if opts:
                    g[bi][li] = self.rng.choice(opts)
        return g

    def _fitness(self, ind, schedule: MatchSchedule, cap_bytes: int) -> float:
        total_bytes = 0
        # Sum tile volumes across tensors using dtype size
        for tname, tiles in schedule.tensor_tiles.items():
            tensor = schedule.tensors.get(tname)
            if tensor is None:
                continue
            itemsize = np.dtype(tensor.dtype).itemsize if hasattr(tensor, "dtype") else 1
            for tt in tiles:
                vol = 1
                for td in tt.tiled_dims:
                    vol *= int(td.size)
                total_bytes += vol * itemsize
        if total_bytes > cap_bytes:
            return 1e9 + (total_bytes - cap_bytes)
        return -float(total_bytes)

    def _apply(self, ind, schedule: MatchSchedule):
        # Map loop choices to dims with same names
        for bi, blk in enumerate(schedule.blocks):
            for li, lp in enumerate(blk.loops):
                chosen = ind[bi][li] if bi < len(ind) and li < len(ind[bi]) else 1
                for tname, tiles in schedule.tensor_tiles.items():
                    for tt in tiles:
                        for td in tt.tiled_dims:
                            if td.dim.name == lp.dim.name:
                                td.size = max(1, min(td.dim.size, chosen))

    def _get_lowest_mem_capacity(self, schedule: MatchSchedule):
        em = getattr(schedule, "exec_module", None)
        if em is None:
            return None
        try:
            mems = em.module_memories()
            if mems:
                return int(mems[0].k_bytes) * 1024
        except Exception:
            return None
        return None
