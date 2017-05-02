from random import randint, random, choice
from itertools import count
import argparse
import time
from statistics import mean, variance
from math import sqrt

from plan import WorkPlan
from logger import TaskLogger

from plan_genetic import make_plan as pl_genetic
from plan_simple import make_plan as pl_simple

TIME_MAX = 100
TASKS_LIMIT = 1  # 0 - no limit


def print_log(level, text):
    if level <= verbosity:
        print(text)

verbosity = 0


class Task:
    _ids = count(0)

    def __init__(self, memory, disk, cpu, time_total):
        self.id = self._ids.__next__()
        self.memory = memory    # 1 - 64
        self.disk = disk        # 0 - 16
        self.cpu = cpu          # 1 - 64
        self.time_total = time_total

        print_log(1, "New task #%d: cpu = %2d, time = %3d" % (self.id, self.cpu, self.time_total))


class TasksPool:
    _tasks_cnt = 0

    def __init__(self):
        self.tasks = []

    def check_new_tasks(self):
        if TASKS_LIMIT != 0 and self._tasks_cnt >= TASKS_LIMIT:
            return False

        if random() < 0.9:
            return False

        cnt = randint(1, 3)
        if cnt == 0:
            return False

        self.add_random_tasks(cnt)
        return True

    def remove_task(self, task):
        self.tasks.remove(task)

    def add_random_tasks(self, n):
        for i in range(n):
            task = Task(
                randint(1, 64),  # 1 - 64
                randint(0, 16),  # 0 - 16
                randint(1, 64),
                randint(10, TIME_MAX),
            )
            self.tasks.append(task)
        self._tasks_cnt += n


class Machine:
    _ids = count(0)

    def __init__(self, memory, disk, cpu, mid=None):
        if mid is None:
            self.id = self._ids.__next__()
        else:
            self.id = mid

        self.memory = self.memory_free = memory
        self.disk = self.disk_free = disk
        self.cpu = self.cpu_free = cpu

        self.workload = []
        self.done = []

        self.idle = True  # Cluster can revoke machine when it's idle
        self.fixed = False  # We always pay for fixed machine, but cluster can't revoke such one

        self.cost = self.cpu
        self.price = 0

    def check_task(self, task):
        if self.cpu_free < task.cpu or self.memory_free < task.memory or self.disk_free < task.disk:
            return False
        else:
            return True

    def check_task_clean(self, task):
        if self.cpu < task.cpu or self.memory < task.memory or self.disk < task.disk:
            return False
        else:
            return True

    def run_task(self, task):
        # if not self.check_task(task):
        #    raise ValueError("Task #%d does not fit machine #%d" % (task.id, self.id))
        self.workload.append({"task": task, "time_left": task.time_total})
        self.update_free_resources()

    def update_free_resources(self):
        cpu = mem = disk = 0
        for task in self.workload:
            cpu += task["task"].cpu
            disk += task["task"].disk
            mem += task["task"].memory

        self.cpu_free = self.cpu - cpu
        self.disk_free = self.disk - disk
        self.memory_free = self.memory - mem

        if len(self.workload) == 0:
            self.idle = True
        else:
            self.idle = False

    def go(self):
        if not self.idle or self.fixed:
            self.price += self.cost

        exited = False
        self.done = []
        for task in self.workload:
            task["time_left"] -= 1
            if task["time_left"] <= 0:
                exited = True
                self.workload.remove(task)
                self.done.append(task)
                self.update_free_resources()
        return exited

    def clone(self):
        new = self.__class__(self.memory, self.disk, self.cpu, mid=self.id)

        new.fixed = self.fixed

        for task in self.workload:
            new.workload.append(task)

        new.update_free_resources()

        return new


class Cluster:
    def __init__(self):
        self.machines = []
        self.time = 0
        self.idle = True

    def make_machines(self):
        fm = Machine(64, 32, 128)
        fm.fixed = True
        self.machines.append(fm)

        fm = Machine(32, 32, 32)
        fm.fixed = True
        self.machines.append(fm)

        fm = Machine(128, 20, 64)
        fm.fixed = True
        self.machines.append(fm)

        fm = Machine(64, 128, 16)
        fm.fixed = True
        self.machines.append(fm)

        return 4

    def add_random_machines(self, n):
        for i in range(n):
            task = Machine(
                randint(1, 16) * 8,
                randint(8, 128),
                randint(2, 32) * 4,
            )
            self.machines.append(task)

    def check_machines(self):
        changed = False

        if random() >= 0.95:
            free = []
            for machine in self.machines:
                if machine.idle and not machine.fixed:
                    free.append(machine)

            if len(free) > 0:
                machine = choice(free)
                self.machines.remove(machine)
                changed = True

        if random() >= 0.95:
            self.add_random_machines(1)
            changed = True

        return changed

    def get_tasks_cnt(self):
        return sum((len(machine.workload) for machine in self.machines))

    def go(self):
        exited = False
        for machine in self.machines:
            exited = exited or machine.go()
        self.idle = all(machine.idle for machine in self.machines)
        self.time += 1
        return exited


class Scheduler:
    def __init__(self, tp, cl):
        self.tp = tp
        self.cluster = cl

        self.logger = TaskLogger()

        self.plan = WorkPlan()
        self.plan_outdated = True

    def run_tasks(self, gen):
        if self.plan_outdated:
            self.plan = gen(self.cluster.machines, self.tp.tasks, self.plan)
            self.plan_outdated = False

        plan = self.plan

        for machine in self.cluster.machines:
            mid = machine.id
            if mid not in plan.chains:
                continue

            chain = plan.chains[mid]

            while len(chain) > 0:
                next_task = None
                next_task_id = plan.chains[mid][0]
                for task in self.tp.tasks:
                    if task.id == next_task_id:
                        next_task = task

                if next_task is None:
                    raise ValueError

                if not machine.check_task(next_task):
                    break

                del chain[0]
                self.tp.remove_task(next_task)

                if machine.idle:
                    self.logger.machine_works(
                        machine_id=machine.id,
                        time=self.cluster.time,
                        resources={"cpu": machine.cpu, "memory": machine.memory, "disk": machine.disk}
                    )

                machine.run_task(next_task)
                self.logger.task_started(
                    task_id=next_task.id,
                    machine_id=machine.id,
                    time=self.cluster.time,
                    resources={"cpu": next_task.cpu, "memory": next_task.memory, "disk": next_task.disk}
                )

    def loop(self, plan_generator):
        ready = True
        while len(self.tp.tasks) > 0 or not cluster.idle:

            if ready:
                print_log(1, "Time: %8d; Tasks cnt: %d + %d" %
                          (self.cluster.time, len(self.tp.tasks), cluster.get_tasks_cnt()))
                self.run_tasks(plan_generator)
                ready = False

            exited = self.cluster.go()

            if exited:
                ready = True
                for machine in self.cluster.machines:
                    for task in machine.done:
                        self.logger.task_done(
                            task_id=task["task"].id,
                            time=self.cluster.time
                        )
                    del machine.done[:]
                    if machine.idle:
                        self.logger.machine_idle(
                            machine_id=machine.id,
                            time=self.cluster.time
                        )

            if self.tp.check_new_tasks():
                ready = True
                self.plan_outdated = True

            if self.cluster.check_machines():
                ready = True
                self.plan_outdated = True

        price = 0
        for machine in self.cluster.machines:
            price += machine.price

        return self.cluster.time, price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment an image using QR markers')

    parser.add_argument('algorithm', help='("simple" or "genetic")')

    parser.add_argument('-t', '--tasks', metavar='T', type=int, default='100', help='Number of tasks')
    parser.add_argument('-m', '--machines', metavar='M', type=int, default='10', help='Number of machines')

    parser.add_argument('-r', '--repeat', metavar='R', type=int, default='1', help='Repeat simulation R times')

    parser.add_argument('-d', '--draw', dest='draw', action='store_true', help='Draw log (ignored when R > 1)')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-vv', '--verbose', dest='verbose2', action='store_true', help='Too verbose mode')

    args = parser.parse_args()

    verbosity = 2 if args.verbose2 else (1 if args.verbose else 0)

    algorithm = None
    if args.algorithm == 'genetic':
        algorithm = pl_genetic
    elif args.algorithm == 'simple':
        algorithm = pl_simple
    else:
        print("Unknown algorithm name")
        exit(1)

    cl_times = []
    costs = []
    times = []
    repeat = args.repeat

    for r in range(repeat):
        task_pool = TasksPool()
        task_pool.add_random_tasks(args.tasks)

        cluster = Cluster()
        m = args.machines - cluster.make_machines()
        if m > 0:
            cluster.add_random_machines(m)

        scheduler = Scheduler(task_pool, cluster)

        start = time.time()
        cl_time, cost = scheduler.loop(algorithm)
        end = time.time()
        print_log(0, "#%3d: Time: %8d; Cost: %8d; Time: %5.2f" % (r, cl_time, cost, end - start))
        cl_times.append(cl_time)
        costs.append(cost)
        times.append(end - start)

        scheduler.logger.dump_yaml('log_' + str(r) + '.txt')
        if repeat == 1 and args.draw:
            scheduler.logger.draw_all()

    print_log(0, "---------------------")
    print_log(0, "Mean cluster time: %8d; mean cost: %8d; mean time: %5.2f" %
              (mean(cl_times), mean(costs), mean(times)))
    print_log(0, "         variance: %8d;  variance: %8d;  variance: %5.2f" %
              (sqrt(variance(cl_times)), sqrt(variance(costs)), sqrt(variance(times))))
