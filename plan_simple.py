from plan import WorkPlan


def make_plan(machines, tasks):
    machines = [machine.clone() for machine in machines]

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
