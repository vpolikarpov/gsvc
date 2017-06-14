from deap import base, creator, tools, algorithms
from random import randint, getrandbits, choice, shuffle, randrange
from plan import WorkPlan, PlanGenerator
from plan_simple import SimpleGenerator
from functools import reduce
from itertools import zip_longest
from time import time as get_time
from operator import attrgetter

from tools import with_next, write_log


def random_plan(cls, machines, tasks):
    wp = cls()
    wp.randomize(machines, tasks)
    return wp


def simple_plan(cls, machines, tasks, scalar_function):
    machines = machines[:]
    shuffle(machines)
    sg = SimpleGenerator(machines, tasks, {}, scalar_function)
    wp = cls(sg.make_one_plan())
    return wp


def make_plan_from_parts(cls, parts):
    for part in parts:
        part["pos"] = part["pos1"] if bool(getrandbits(1)) else part["pos2"]
    parts.sort(key=lambda p: p["pos"][1])
    parts.sort(key=lambda p: p["pos"][0])

    child = cls()
    last_chain = -1
    candidates = []
    for part, nxt in with_next(parts):
        if part["pos"][0] != last_chain:
            child.create_chain(part["pos"][0])
            last_chain = part["pos"][0]

        candidates.append(part["data"])

        if nxt is None or part["pos"] != nxt["pos"]:
            shuffle(candidates)
            for c in candidates:
                child.append_to_chain(part["pos"][0], c)
            candidates = []

    return child


def mate(cls, plan1, plan2):
    parts = []

    for mid, chain in plan1.chains.items():
        if len(chain) == 0:
            continue

        tid = chain[0]
        part_data = [tid]
        part_pos_1 = (mid, 0)
        part_pos_2 = plan2.get_task_pos(tid)

        for idx, tid in enumerate(chain[1:]):
            if tid == plan2.get_next_task_id(part_data[-1]):
                part_data.append(tid)
            else:
                parts.append({
                    "data": part_data,
                    "pos1": part_pos_1,
                    "pos2": part_pos_2,
                })
                part_data = [tid]
                part_pos_1 = (mid, idx + 1)
                part_pos_2 = plan2.get_task_pos(tid)

        parts.append({
            "data": part_data,
            "pos1": part_pos_1,
            "pos2": part_pos_2,
        })

    ch1, ch2 = make_plan_from_parts(cls, parts), make_plan_from_parts(cls, parts)

    if ch1.count_tasks() != ch2.count_tasks() or ch1.count_tasks() != plan1.count_tasks():
        raise RuntimeError("Bad children")

    return ch1, ch2


def mutate(machines, tasks, plan, max_exchanges, max_moves):
    exchanges = randint(1, max_exchanges) if max_exchanges > 1 else 1
    moves = randint(1, max_moves) if max_moves > 1 else 1

    n = plan.count_tasks()
    if n < 2:
        return plan,

    for i in range(exchanges):

        # Select and find first task
        tid_1, mid_1, idx_1 = plan.get_random_task()
        chain_1 = plan.chains[mid_1]

        machine_1 = None
        for m in machines:
            if m.id == mid_1:
                machine_1 = m
                break

        task_1 = None
        for t in tasks:
            if t.id == tid_1:
                task_1 = t
                break

        if machine_1 is None or task_1 is None:
            raise ValueError

        # Get jobs from machines that can contain task_1
        m_fit = [m for m in machines if m.check_task_clean(task_1)]
        chains_fit = [plan.chains[m.id] for m in m_fit if m.id in plan.chains]
        tid_fit = [task for chain in chains_fit for task in chain]

        # Remove self id from exchange list
        tid_fit.remove(tid_1)

        # Remove tasks that doesn't fit to first machine
        tid_fit_2 = []
        for tid in tid_fit:
            for t in tasks:
                if t.id == tid and machine_1.check_task_clean(t):
                    tid_fit_2.append(tid)

        if len(tid_fit_2) == 0:
            continue
            # TODO: Should i retry exchange?

        # Find second item
        tid_2 = choice(tid_fit_2)
        mid_2, idx_2 = plan.find_task(tid_2)
        chain_2 = plan.chains[mid_2]

        # Replace
        chain_1[idx_1] = tid_2
        chain_2[idx_2] = tid_1

        # Clear fitness caches
        if mid_1 in plan.fitness_cache:
            del plan.fitness_cache[mid_1]
        if mid_2 in plan.fitness_cache:
            del plan.fitness_cache[mid_2]

    for i in range(moves):

        # Select and find task
        tid_1, mid_1, idx_1 = plan.get_random_task()
        chain_1 = plan.chains[mid_1]

        task_1 = None
        for t in tasks:
            if t.id == tid_1:
                task_1 = t
                break

        if task_1 is None:
            raise ValueError

        # Select destination chain
        m_ids = [m.id for m in machines if m.check_task_clean(task_1)]
        mid_2 = choice(m_ids)
        if mid_2 in plan.chains:
            chain_2 = plan.chains[mid_2]
        else:
            chain_2 = plan.create_chain(mid_2)

        # Select exact destination
        n = len(chain_2)
        idx_2 = randrange(n) if n > 0 else 0

        # Move
        del chain_1[idx_1]
        chain_2.insert(idx_2, tid_1)

        # Clear fitness caches
        if mid_1 in plan.fitness_cache:
            del plan.fitness_cache[mid_1]
        if mid_2 in plan.fitness_cache:
            del plan.fitness_cache[mid_2]

    return plan,


def evaluate(machines_all, tasks, scalar_function, plan):
    time, price = plan.evaluate(machines_all, tasks)
    return scalar_function(time, price),


def select_tournament_unique(individuals, k, tournsize):
    individuals = individuals[:]
    chosen = []
    for i in range(k):
        aspirants = []
        while len(aspirants) < tournsize:
            new = choice(individuals)
            if new not in aspirants:
                aspirants.append(new)

        ind = max(aspirants, key=attrgetter("fitness"))
        chosen.append(ind)
        individuals.remove(ind)
    return chosen


def expand_plan(machines, tasks, plan, new_tasks):
    machines = [machine.clone() for machine in machines]
    machines = [m for m in machines if m.fixed] + [m for m in machines if not m.fixed]
    tasks = tasks[:]

    try:
        if len(new_tasks) == 0:
            return
    except TypeError:
        new_tasks = [new_tasks]

    ind = {}
    for machine in machines:
        mid = machine.id
        ind[mid] = 0

    while True:

        for machine in machines:

            machine.reserve()
            mid = machine.id

            next_task = None
            try:
                next_task_id = plan.chains[mid][ind[mid]]
                for task in tasks:
                    if task.id == next_task_id:
                        next_task = task
            except (IndexError, KeyError):
                pass

            while next_task is not None:
                if machine.check_task(next_task):
                    machine.run_task(next_task)
                    ind[mid] += 1

                    next_task = None
                    try:
                        next_task_id = plan.chains[mid][ind[mid]]
                    except IndexError:
                        break
                    for task in tasks:
                        if task.id == next_task_id:
                            next_task = task
                else:
                    break

            if next_task is None:
                while len(new_tasks) > 0:
                    new_task = new_tasks[0]
                    if machine.check_task(new_task):
                        machine.run_task(new_task)
                        plan.append_to_chain(mid, new_task.id)
                        if mid in plan.fitness_cache:
                            del plan.fitness_cache[mid]
                        new_tasks.pop(0)
                    else:
                        break

        if len(new_tasks) == 0:
            break

        exited = False
        while not exited:
            for machine in machines:
                exited = machine.go() or exited

    if not plan.check_plan(machines, tasks):
        raise RuntimeError("Bad plan")
    return


class GeneticGenerator (PlanGenerator):
    def __init__(self, machines, tasks, settings, func):
        super().__init__(machines, tasks, settings, func)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("WorkPlan", WorkPlan, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", simple_plan, creator.WorkPlan, self.machines, self.tasks, self.scalar)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", mate, creator.WorkPlan)
        self.toolbox.register("mutate", mutate, self.machines, self.tasks, max_exchanges=1, max_moves=1)
        self.toolbox.register("select", select_tournament_unique, tournsize=3)
        self.toolbox.register("evaluate", evaluate, self.machines, self.tasks, self.scalar)

        self.settings["GEN"] = self.settings.get("GEN", 100)  # generations number
        self.settings["MU"] = self.settings.get("MU", 10)  # generation size
        self.settings["LAMBDA"] = self.settings.get("LAMBDA", 20)  # offspring size
        self.settings["C_PROBABILITY"] = self.settings.get("CX_PROBABILITY", 0.4)  # crossing over
        self.settings["M_PROBABILITY"] = self.settings.get("generations", 0.4)  # mutation

        self.continuous = getattr(self.settings, "continuous", False)
        if self.continuous:
            self.population = self.toolbox.population(n=self.settings["MU"])

    def get_plan(self):
        self.machines[:] = [machine.clone() for machine in self.machines_external]

        if not self.continuous:
            self.population = self.toolbox.population(n=self.settings["MU"])

        self.check_population()

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(key=lambda ind: ind.fitness)
        stats.register("min", reduce, lambda x, y: x if x.dominates(y) else y)

        cpu_time = get_time()

        s = self.settings
        self.population, logbook = algorithms.eaMuPlusLambda(self.population, self.toolbox,
                                                             s["MU"], s["LAMBDA"],
                                                             s["C_PROBABILITY"], s["M_PROBABILITY"], s["GEN"],
                                                             halloffame=hof, stats=stats, verbose=False)

        cpu_time = get_time() - cpu_time

        f_init = logbook[0]['min'].values
        f_res = logbook[self.settings["GEN"]]['min'].values

        prct = [(fi - fr) * 100 / fi if fi > 0 else 0 for fi, fr in zip_longest(f_init, f_res)]

        write_log(2, "%6.2f [cnt: %4d; cpu_time: %7.2f]" % (prct[0], len(self.tasks), cpu_time))

        if prct[0] < 0:
            write_log(1, "Regression detected")

        self.check_population()

        return hof.items[0]

    def remove_task(self, task):
        if self.continuous:
            self.check_population()

        super().remove_task(task)

        if self.continuous:
            for plan in self.population:
                task_pos = plan.find_task(task.id)
                if task_pos is None:
                    continue
                del plan.get_chain(task_pos[0])[task_pos[1]]
                try:
                    del plan.fitness_cache[task_pos[0]]
                except KeyError:
                    pass
                del plan.fitness.values

            self.check_population()

    def add_task(self, task):
        if self.continuous:
            self.check_population()

        super().add_task(task)

        if self.continuous:
            for plan in self.population:
                expand_plan(self.machines, self.tasks, plan, task)
                del plan.fitness.values

            self.check_population()

    def remove_machine(self, machine):
        if self.continuous:
            self.check_population()

        super().remove_machine(machine)

        if self.continuous:
            mid = machine.id
            for plan in self.population:
                if mid in plan.chains:
                    task_ids = plan.chains[mid]
                    del plan.chains[mid]
                    try:
                        del plan.fitness_cache[mid]
                    except KeyError:
                        pass
                    tasks = []
                    for tid in task_ids:
                        for task in self.tasks:
                            if task.id == tid:
                                tasks.append(task)
                    if tasks:
                        expand_plan(self.machines, self.tasks, plan, tasks)
                        del plan.fitness.values

        if self.continuous:
            self.check_population()

    def add_machine(self, machine):
        if self.continuous:
            self.check_population()

        super().add_machine(machine)

        if self.continuous:
            self.check_population()

    def check_population(self):
        for plan in self.population:
            if not plan.check_plan(self.machines, self.tasks):
                raise RuntimeError("Population is broken")
