from random import randint, random, choice, triangular
from itertools import count
import argparse
import time
from statistics import mean, variance
from math import sqrt, ceil

from plan import WorkPlan
from logger import TaskLogger
from tools import write_log, set_verbosity

from plan_genetic import GeneticGenerator
from plan_simple import SimpleGenerator

import yaml

TIME_MAX = 100


class Task:
    _ids = count(0)

    def __init__(self, memory, disk, cpu, time_total):
        self.id = self._ids.__next__()
        self.memory = memory    # 1 - 64
        self.disk = disk        # 0 - 16
        self.cpu = cpu          # 1 - 64
        self.time_estimated = time_total
        self.time_total = time_total

        write_log(2, "New task #%d: cpu = %2d, time = %3d" % (self.id, self.cpu, self.time_total))

    def __str__(self):
        return "Task #%d" % self.id

    def choose_length(self, timings):
        self.time_total = ceil(self.time_total * triangular(**timings))


class TasksPool:
    _tasks_cnt = 0

    def __init__(self, limit, timings):
        self.tasks = []
        self.limit = limit
        self.timings = timings

    def check_new_tasks(self):
        if self.limit != 0 and self._tasks_cnt >= self.limit:
            return []

        if random() < 0.9:
            return []

        cnt = randint(1, 3)
        if cnt == 0:
            return []

        new = self.add_random_tasks(cnt)
        return new

    def remove_task(self, task):
        self.tasks.remove(task)

    def add_random_tasks(self, n):
        new_tasks = []
        for i in range(n):
            task = Task(
                randint(1, 64),  # 1 - 64
                randint(0, 16),  # 0 - 16
                randint(1, 64),
                randint(10, TIME_MAX),
            )
            task.choose_length(self.timings)
            new_tasks.append(task)
        self._tasks_cnt += n
        self.tasks += new_tasks
        return new_tasks


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

        self.idle = True
        self.reserved = False  # Cluster can't revoke machine when it's reserved
        self.fixed = False  # We always pay for fixed machine, but they are always reserved

        self.cost = ceil(self.cpu + (self.memory / 2) + (self.disk / 10))
        self.credit_period = 1

        self.price = 0
        self.paid_time = 0

    def __str__(self):
        return "Machine #%d" % self.id

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
        if not self.reserved:
            raise ValueError("You should reserve machine first")

        # if not self.check_task(task):
        #    raise ValueError("Task #%d does not fit machine #%d" % (task.id, self.id))

        self.workload.append({"task": task, "time_left": task.time_total})
        self.update_free_resources()

    def reserve(self):
        self.reserved = True

    def free(self):
        if self.fixed:
            raise ValueError("You cannot free fixed machine")
        else:
            self.reserved = False

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
        if self.reserved:
            if self.paid_time <= 0:
                self.price += self.cost * self.credit_period
                self.paid_time = self.credit_period
            self.paid_time -= 1

        exited = False
        self.done = []
        for task in self.workload:
            task["time_left"] -= 1
            if task["time_left"] <= 0:
                exited = True
                self.done.append(task)

        if exited:
            for task in self.done:
                self.workload.remove(task)
            self.update_free_resources()

        return exited

    def clone(self):
        new = self.__class__(self.memory, self.disk, self.cpu, mid=self.id)

        new.fixed = self.fixed
        new.reserved = self.reserved

        new.credit_period = self.credit_period

        for task in self.workload:
            new.workload.append(task.copy())

        new.update_free_resources()

        return new


class Cluster:
    def __init__(self, limit, credit_period=1):
        self.machines = []
        self.time = 0
        self.idle = True
        self.limit = limit
        self.credit_period = credit_period

    def add_machine(self, machine):
        machine.credit_period = self.credit_period
        self.machines.append(machine)

    def add_random_machines(self, n):
        new_machines = []
        for i in range(n):
            machine = Machine(
                randint(1, 16) * 8,
                randint(8, 128),
                randint(2, 32) * 4,
            )
            machine.credit_period = self.credit_period
            new_machines.append(machine)
        self.machines += new_machines
        return new_machines

    def check_deleted_machines(self):
        deleted_machines = []

        if random() >= 0.98:
            free = []
            for machine in self.machines:
                if not machine.reserved and not machine.fixed:
                    free.append(machine)

            if len(free) > 0:
                machine = choice(free)
                self.machines.remove(machine)
                deleted_machines.append(machine)

        return deleted_machines

    def check_new_machines(self):
        if self.limit != 0 and len(self.machines) >= self.limit:
            return []

        new_machines = []

        if random() >= 0.98:
            new_machines += self.add_random_machines(1)

        return new_machines

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
    def __init__(self, tp, cl, gen):
        self.tp = tp
        self.cluster = cl

        self.logger = TaskLogger()

        self.plan = WorkPlan()
        self.plan_outdated = True

        self.generator = gen

        for machine in self.cluster.machines:
            if machine.fixed and not machine.reserved:
                self.logger.machine_works(
                    machine_id=machine.id,
                    time=self.cluster.time,
                    resources={"cpu": machine.cpu, "memory": machine.memory, "disk": machine.disk}
                )
                machine.reserve()

    def run_tasks(self):
        if self.plan_outdated:
            self.plan = self.generator.get_plan()
            self.plan_outdated = False

        plan = self.plan

        for machine in self.cluster.machines:
            mid = machine.id
            if mid not in plan.chains:
                continue

            chain = plan.chains[mid]

            while len(chain) > 0:
                next_task = None
                next_task_id = chain[0]
                for task in self.tp.tasks:
                    if task.id == next_task_id:
                        next_task = task

                if next_task is None:
                    raise ValueError

                if not machine.check_task(next_task):
                    break

                del chain[0]
                self.tp.remove_task(next_task)
                self.generator.remove_task(next_task)

                if not machine.reserved:
                    self.logger.machine_works(
                        machine_id=machine.id,
                        time=self.cluster.time,
                        resources={"cpu": machine.cpu, "memory": machine.memory, "disk": machine.disk}
                    )
                    machine.reserve()

                machine.run_task(next_task)
                self.logger.task_started(
                    task_id=next_task.id,
                    machine_id=machine.id,
                    time=self.cluster.time,
                    resources={"cpu": next_task.cpu, "memory": next_task.memory, "disk": next_task.disk}
                )

    def loop(self):
        ready = True
        while len(self.tp.tasks) > 0 or not cluster.idle:

            if ready:
                write_log(1, "Time: %8d; Tasks cnt: %4d; Pending: %4d" %
                          (self.cluster.time, cluster.get_tasks_cnt(), len(self.tp.tasks)), one_line=True)
                self.run_tasks()
                ready = False

            exited = self.cluster.go()

            if exited:
                ready = True
                self.plan_outdated = True
                for machine in self.cluster.machines:
                    for task in machine.done:
                        self.logger.task_done(
                            task_id=task["task"].id,
                            time=self.cluster.time
                        )
                    del machine.done[:]
                    if machine.idle and not machine.fixed:
                        if machine.id not in self.plan.chains or len(self.plan.chains[machine.id]) == 0:
                            machine.free()
                            self.logger.machine_idle(
                                machine_id=machine.id,
                                time=self.cluster.time
                            )

            deleted_machines = self.cluster.check_deleted_machines()
            new_machines = self.cluster.check_new_machines()
            new_tasks = self.tp.check_new_tasks()

            if new_tasks or new_machines or deleted_machines:
                ready = True
                self.plan_outdated = True

            if deleted_machines:
                [self.generator.remove_machine(machine) for machine in deleted_machines]
                write_log(1, "Time: %8d; Deleted machine!" % self.cluster.time)

            if new_machines:
                [self.generator.add_machine(machine) for machine in new_machines]
                write_log(1, "Time: %8d; New machine!" % self.cluster.time)

            if new_tasks:
                [self.generator.add_task(task) for task in new_tasks]
                write_log(1, "Time: %8d; New tasks!" % self.cluster.time)

        price = 0
        for machine in self.cluster.machines:
            price += machine.price

        return self.cluster.time, price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment an image using QR markers')

    parser.add_argument('algorithm', help='("simple" or "genetic")', nargs='?', default="simple")

    parser.add_argument('-C', '--config', help='Configuration file')

    parser.add_argument('-c', '--continuous', dest='genetic_continuous', action='store_true')

    parser.add_argument('-t', '--tasks', metavar='T', type=int, default='100', help='Number of tasks')
    parser.add_argument('-m', '--machines', metavar='M', type=int, default='10', help='Number of machines')

    parser.add_argument('-r', '--repeat', metavar='R', type=int, default='1', help='Repeat simulation R times')

    parser.add_argument('-d', '--draw', dest='draw', action='store_true', help='Draw log (ignored when R > 1)')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-vv', '--verbose', dest='verbose2', action='store_true', help='Too verbose mode')

    args = parser.parse_args()

    if args.config:
        conf = {}
        with open(args.config, "r") as f:
            conf_text = f.read()
            conf = yaml.load(conf_text)

        machines_info = conf.get("machines", {})
        credit_period = machines_info.get("credit_period", 1)
        machines_own = machines_info.get("own", [])
        machines_remote = machines_info.get("remote", {})
        machines_cnt = machines_remote.get("initial", 10)
        machines_limit = machines_remote.get("limit", 0)

        tasks_info = conf.get("tasks", {})
        tasks_cnt = tasks_info.get("initial", 100)
        tasks_limit = tasks_info.get("limit", 0)
        tasks_timings = tasks_info.get("timings", {"low": 0.5, "high": 1, "mode": 0.75})

        repeat = conf.get("repeat", 1)
        verbosity = conf.get("verbosity", 0)
        draw = conf.get("draw", False)

        alg_info = conf["algorithm"]
        alg_name = alg_info["name"]
        alg_settings = alg_info.get("settings", {})

    else:
        credit_period = 1
        machines_own = [
            {"memory": 64, "disk": 32, "cpu": 128},
            {"memory": 32, "disk": 32, "cpu": 32},
            {"memory": 128, "disk": 20, "cpu": 64},
            {"memory": 64, "disk": 128, "cpu": 16},
        ]
        machines_cnt = args.machines
        machines_limit = 0

        tasks_cnt = args.tasks
        tasks_limit = 1
        tasks_timings = {"low": 0.5, "high": 1, "mode": 0.75}

        repeat = args.repeat
        verbosity = 2 if args.verbose2 else (1 if args.verbose else 0)
        draw = args.draw

        alg_name = args.algorithm
        alg_settings = {}
        if args.genetic_continuous:
            alg_settings["continuous"] = True

    if repeat > 1 and draw is True:
        raise RuntimeError("Cannot draw result when repeating multiple times")

    set_verbosity(verbosity)

    cl_times = []
    costs = []
    times = []

    for r in range(repeat):
        task_pool = TasksPool(limit=tasks_limit, timings=tasks_timings)
        task_pool.add_random_tasks(tasks_cnt)

        cluster = Cluster(limit=machines_limit, credit_period=credit_period)

        for m_info in machines_own:
            fm = Machine(m_info["memory"], m_info["disk"], m_info["cpu"])
            fm.fixed = True
            cluster.add_machine(fm)

        if machines_cnt > 0:
            cluster.add_random_machines(machines_cnt)

        generator = None
        if alg_name == 'genetic':
            generator = GeneticGenerator(cluster.machines, task_pool.tasks, alg_settings)
        elif alg_name == 'simple':
            generator = SimpleGenerator(cluster.machines, task_pool.tasks, alg_settings)
        else:
            print("Unknown algorithm name")
            exit(1)

        scheduler = Scheduler(task_pool, cluster, generator)

        start = time.time()
        cl_time, cost = scheduler.loop()
        end = time.time()
        write_log(0, "#%3d: Time: %8d; Cost: %8d; Time: %5.2f" % (r, cl_time, cost, end - start))
        cl_times.append(cl_time)
        costs.append(cost)
        times.append(end - start)

        scheduler.logger.dump_yaml('log_' + str(r) + '.txt')
        if repeat == 1 and draw:
            scheduler.logger.draw_all(credit_period=credit_period)

    if repeat > 1:
        write_log(0, "---------------------")
        write_log(0, "Mean cluster time: %8d; mean cost: %8d; mean time: %5.2f" %
                  (mean(cl_times), mean(costs), mean(times)))
        write_log(0, "         variance: %8d;  variance: %8d;  variance: %5.2f" %
                  (sqrt(variance(cl_times)), sqrt(variance(costs)), sqrt(variance(times))))
