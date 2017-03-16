from deap import base, creator, tools, algorithms
from random import randint, getrandbits, choice, shuffle
from plan import WorkPlan
from plan_simple import make_plan as pl_simple


def with_next(iterable):
    iterator = iter(iterable)
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (current_item, next_item)
        current_item = next_item
    yield (current_item, None)


def random_plan(cls, machines, tasks):
    wp = cls()
    wp.randomize(machines, tasks)
    return wp


def pseudo_random_plan(cls, machines, tasks):
    machines = machines[:]
    shuffle(machines)
    wp = cls(pl_simple(machines, tasks))
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
                part_pos_1 = (mid, idx)
                part_pos_2 = plan2.get_task_pos(tid)

        parts.append({
            "data": part_data,
            "pos1": part_pos_1,
            "pos2": part_pos_2,
        })

    ch1, ch2 = make_plan_from_parts(cls, parts), make_plan_from_parts(cls, parts)

    if ch1.count_tasks() != ch2.count_tasks() or ch1.count_tasks() != plan1.count_tasks():
        print("ВАСЯ ЕСТ НЕНАША")

    return ch1, ch2


def mutate(machines, tasks, plan, max_exchanges):
    exchanges = randint(1, max_exchanges) if max_exchanges > 1 else 1

    n = plan.count_tasks()
    if n < 2:
        return plan,

    for i in range(exchanges):

        # Select and find first task
        n1 = randint(1, n)
        for mid, chain in plan.chains.items():
            for idx, tid in enumerate(chain):
                n1 -= 1
                if n1 == 0:
                    chain_1 = chain
                    tid_1 = tid
                    idx_1 = idx
                    mid_1 = mid

        machine = None
        for m in machines:
            if m.id == mid_1:
                machine = m
                break

        task = None
        for t in tasks:
            if t.id == tid_1:
                task = t
                break

        # Get jobs from machines that can contain first task
        m_fit = [m for m in machines if m.check_task_clean(task)]
        chains_fit = [plan.chains[m.id] for m in m_fit if m.id in plan.chains]
        tid_fit = [task for chain in chains_fit for task in chain]

        # Remove self id from exchange list
        tid_fit.remove(tid_1)

        # Remove tasks that doesn't fit to first machine
        tid_fit_2 = []
        for tid in tid_fit:
            for t in tasks:
                if t.id == tid and machine.check_task_clean(t):
                    tid_fit_2.append(tid)

        if len(tid_fit_2) == 0:
            continue
            # TODO: Should i retry exchange?

        # Find second item
        tid_2 = choice(tid_fit_2)

        for _, chain in plan.chains.items():
            for idx, tid in enumerate(chain):
                if tid_2 == tid:
                    chain_2 = chain
                    idx_2 = idx

        # Replace
        chain_1[idx_1] = tid_2
        chain_2[idx_2] = tid_1

    return plan,


def evaluate(machines, tasks, plan):

    # Clone active machines
    machines = [machine.clone() for machine in machines if machine.id in plan.chains]

    max_time = 0
    cost = 0

    for machine in machines:
        mid = machine.id

        next_task = None
        next_task_id = plan.chains[machine.id][0]
        for task in tasks:
            if task.id == next_task_id:
                next_task = task

        i = 0
        time = 0

        while True:

            # Trying to run as more jobs as possible
            while next_task is not None:
                if machine.check_task(next_task):
                    machine.run_task(next_task)
                    i += 1

                    next_task = None
                    try:
                        next_task_id = plan.chains[mid][i]
                    except IndexError:
                        break
                    for task in tasks:
                        if task.id == next_task_id:
                            next_task = task
                else:
                    break

            # Go
            if not machine.idle:
                while not machine.go():
                    time += 1

            if machine.idle and next_task is None:
                max_time = max(max_time, time)
                if not machine.fixed:  # Cost for fixed machines depends from max_time, so we calculate it later
                    cost += machine.cost * time
                break

    # Cost for fixed machines
    for machine in machines:
        if machine.fixed:
            cost += machine.cost * max_time

    return max_time, cost


def make_plan(machines, tasks):

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("WorkPlan", WorkPlan, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", pseudo_random_plan, creator.WorkPlan, machines, tasks)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", mate, creator.WorkPlan)
    toolbox.register("mutate", mutate, machines, tasks, max_exchanges=1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, machines, tasks)

    NGEN = 20
    MU = 10
    LAMBDA = 20
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, halloffame=hof, verbose=False)

    return hof.items[0]
