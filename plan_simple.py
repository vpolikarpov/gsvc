from plan import WorkPlan
from random import shuffle

TRIES = 10


def make_plan(machines, tasks, _):
    selected_plan = None
    min_cost = -1
    min_time = -1

    machines = machines[:]
    for i in range(TRIES):
        shuffle(machines)
        plan = make_one_plan(machines, tasks)
        time, cost = plan.evaluate(machines, tasks)
        if selected_plan is None or (time < min_time and cost < min_cost):
            min_time, min_cost = time, cost
            selected_plan = plan
    return selected_plan


def make_one_plan(machines, tasks):
    machines = [machine.clone() for machine in machines]
    machines = [m for m in machines if m.fixed] + [m for m in machines if not m.fixed]

    tasks = tasks[:]

    plan = WorkPlan()

    while True:
        used = []
        for task in tasks:
            for machine in machines:
                if machine.check_task(task):
                    machine.run_task(task)
                    plan.append_to_chain(machine.id, task.id)
                    used.append(task)
                    break

        tasks = [t for t in tasks if t not in used]

        exited = False
        while not exited:
            for machine in machines:
                exited = machine.go() or exited

        if len(tasks) == 0:
            break

    return plan
