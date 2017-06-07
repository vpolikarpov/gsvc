from random import randrange
from math import ceil

from tools import write_log

from time import time as get_time
time_1 = time_2 = time_3 = 0


class WorkPlan:
    def __init__(self, plan=None):
        self.chains = {}
        self.fitness_cache = {}
        self.time = 0
        self.cost = 0
        if plan:
            for mid, chain in plan.chains.items():
                self.append_to_chain(mid, chain[:])

    def get_next_task_id(self, cur_id):
        for mid, chain in self.chains.items():
            found = False
            for tid in chain:
                if found:
                    return tid
                if tid == cur_id:
                    found = True
        return -1

    def get_task_pos(self, cur_id):
        for mid, chain in self.chains.items():
            for idx, tid in enumerate(chain):
                if tid == cur_id:
                    return mid, idx
        return -1, -1

    def get_random_task(self):
        n = self.count_tasks()
        n1 = randrange(n)
        for mid, chain in self.chains.items():
            for idx, tid in enumerate(chain):
                if n1 == 0:
                    return tid, mid, idx
                n1 -= 1

    def find_task(self, task_id):
        for mid, chain in self.chains.items():
            for idx, tid in enumerate(chain):
                if task_id == tid:
                    return mid, idx

    def create_chain(self, machine_id):
        self.chains[machine_id] = []
        return self.chains[machine_id]

    def get_chain(self, machine_id):
        if machine_id in self.chains:
            return self.chains[machine_id]
        else:
            return None

    def append_to_chain(self, machine_id, tasks):
        chain = self.get_chain(machine_id)
        if chain is None:
            chain = self.create_chain(machine_id)
        try:
            chain += tasks
        except TypeError:
            chain += [tasks]

    def count_tasks(self):
        return sum(len(chain) for _, chain in self.chains.items())

    def check_plan(self, machines, tasks):

        # Check tasks number
        if self.count_tasks() != len(tasks):
            write_log(0, "Plan Error: Wrong tasks number")
            return False

        # Check tasks duplicates
        tids_plain = [tid for _, chain in self.chains.items() for tid in chain]
        for id_1, tid_1 in enumerate(tids_plain):
            for id_2 in range(id_1 + 1, len(tids_plain)):
                tid_2 = tids_plain[id_2]
                if tid_1 == tid_2:
                    write_log(0, "Plan Error: Task duplication")
                    return False

        # Check that each task fits to corresponding machine
        for mid, chain in self.chains.items():
            machine = None
            for m in machines:
                if m.id == mid:
                    machine = m

            if machine is None:
                write_log(0, "Plan Error: nonexistent machine")
                return False

            for tid in chain:
                task = None
                for t in tasks:
                    if t.id == tid:
                        task = t

                if not machine.check_task_clean(task):
                    write_log(0, "Plan Error: Misfit")
                    return False

        return True

    def evaluate(self, machines_all, tasks):
        # Clone active machines
        machines = [machine for machine in machines_all if machine.id in self.chains]

        max_time = 0
        sum_cost = 0

        global time_1, time_2, time_3

        for machine in machines:

            mid = machine.id

            i = 0
            time = 0
            cost = 0

            res_release = []
            res = {
                "cpu": machine.cpu_free,
                "memory": machine.memory_free,
                "disk": machine.disk_free,
            }

            for task in machine.workload:
                res_release.append({"time": task["time_left"], "res": {
                    "cpu": task["task"].cpu,
                    "memory": task["task"].memory,
                    "disk": task["task"].disk,
                }})
                res_release.sort(key=lambda x: x["time"])

            if mid in self.fitness_cache:
                time, cost = self.fitness_cache[mid]
            else:
                next_task = None
                if len(self.chains[mid]) > 0:
                    next_task_id = self.chains[mid][0]
                    for task in tasks:
                        if task.id == next_task_id:
                            next_task = task

                while True:

                    time_0 = get_time()

                    # Trying to run as more jobs as possible
                    while next_task is not None:
                        if next_task.cpu > res["cpu"] or next_task.memory > res["memory"] or next_task.disk > res["disk"]:
                            break
                        else:
                            i += 1

                            j = 0
                            j_max = len(res_release)
                            time_end = time + next_task.time_total
                            while j < j_max and res_release[j]["time"] < time_end:
                                j += 1
                            res_release.insert(j, {
                                "time": time_end,
                                "res": {
                                    "cpu": next_task.cpu,
                                    "memory": next_task.memory,
                                    "disk": next_task.disk,
                                }
                            })

                            res["cpu"] -= next_task.cpu
                            res["memory"] -= next_task.memory
                            res["disk"] -= next_task.disk

                            next_task = None
                            try:
                                next_task_id = self.chains[mid][i]
                            except IndexError:
                                break
                            for task in tasks:
                                if task.id == next_task_id:
                                    next_task = task

                    time_1 += get_time() - time_0
                    time_0 = get_time()

                    # Go
                    try:
                        time = res_release[0]["time"]
                        res["cpu"] += res_release[0]["res"]["cpu"]
                        res["memory"] += res_release[0]["res"]["memory"]
                        res["disk"] += res_release[0]["res"]["disk"]

                        del res_release[0]
                        idle = False
                    except IndexError:
                        idle = True

                    time_2 += get_time() - time_0
                    time_0 = get_time()

                    if idle and next_task is None:
                        time = ceil(time / machine.credit_period) * machine.credit_period
                        cost = machine.cost * time
                        self.fitness_cache[mid] = (time, cost)
                        break

            max_time = max(max_time, time)
            if not machine.fixed:  # Cost for fixed machines depends from max_time, so we calculate it later
                sum_cost += cost

        # Cost for fixed machines
        for machine in machines_all:
            if machine.fixed:
                sum_cost += machine.cost * max_time
        # Should we assume zero cost for fixed machines?

        self.time = max_time
        self.cost = sum_cost

        return max_time, sum_cost

    def randomize(self, machines, tasks):
        for task in tasks:
            ma_list = []  # available machines
            for machine in machines:
                if machine.check_task_clean(task):
                    ma_list.append(machine)
            if len(ma_list) == 1:
                mid = ma_list[0].id
            else:
                mid = ma_list[randrange(len(ma_list))].id

            self.append_to_chain(mid, task.id)


def print_time():
    print("%f, %f, %f" % (time_1, time_2, time_3))


class PlanGenerator:

    def __init__(self, machines, tasks, settings, func):
        self.machines_external = machines[:]
        self.machines = [machine.clone() for machine in self.machines_external]

        self.tasks = []
        self.tasks_pending = tasks[:]
        self.check_pending_tasks()

        self.settings = settings
        self.scalar = func

    def check_pending_tasks(self):
        fit = []
        for task in self.tasks_pending:
            for machine in self.machines:
                if machine.check_task_clean(task):
                    fit.append(task)
                    break

        for task in fit:
            self.tasks_pending.remove(task)
            self.tasks.append(task)

    def get_plan(self):
        return WorkPlan()

    def add_task(self, task):
        write_log(2, " -- Add task # %d" % task.id)
        self.tasks_pending.append(task)
        self.check_pending_tasks()

    def remove_task(self, task):
        write_log(2, " -- Remove task # %d" % task.id)
        self.tasks.remove(task)

    def add_machine(self, machine):
        write_log(2, " -- Add machine # %d" % machine.id)
        self.machines_external.append(machine)
        self.machines.append(machine.clone())
        self.check_pending_tasks()

    def remove_machine(self, machine):
        write_log(2, " -- Remove machine # %d" % machine.id)
        self.machines_external.remove(machine)
        self.machines[:] = [machine.clone() for machine in self.machines_external]
        # There is no simple way to just remove cloned machine
        self.check_pending_tasks()
