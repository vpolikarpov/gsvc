from random import randint, randrange, random, choice, triangular
from itertools import count
import argparse
import time
from statistics import mean, variance
from math import sqrt, ceil, exp
from os import path

from plan import WorkPlan
from logger import TaskLogger
from tools import write_log, set_verbosity

from plan_genetic import GeneticGenerator
from plan_simple import SimpleGenerator

import yaml


class Task:
    _ids = count(0)

    def __init__(self, memory, disk, cpu, time_total):
        self.id = self._ids.__next__()
        self.memory = memory
        self.disk = disk
        self.cpu = cpu
        self.time_estimated = time_total
        self.time_total = time_total

        write_log(2, "New task #%d: cpu = %2d, time = %3d" % (self.id, self.cpu, self.time_total))

    def __str__(self):
        return "Task #%d" % self.id

    def choose_length(self, timings):
        self.time_total = ceil(self.time_total * triangular(**timings))


class TasksPool:
    _tasks_cnt = 0

    def __init__(self, cnf):
        self.tasks = []
        self.additions = {}
        self.conf = cnf
        self.end = False
        if cnf["spawn"]["type"] == "random":
            self.spawn = self.new_tasks_random
        if cnf["spawn"]["type"] == "waves":
            self.spawn = self.new_tasks_waves
            self.conf["spawn"]["last_wave"] = max(wave["time"] for wave in self.conf["spawn"]["waves"])
        else:
            raise ValueError("Tasks spawning type is not set")

    def check_new_tasks(self, time, initial=False):
        tasks = self.spawn(time, initial)
        for task in tasks:
            self.additions[task.id] = time
        return tasks

    def new_tasks_random(self, _, initial):
        s = self.conf["spawn"]

        if initial:
            initial_cnt = s.get("initial-cnt", 0)
            return self.create_tasks(initial_cnt)

        limit = s.get("limit", 0)

        if limit != 0 and self._tasks_cnt >= limit:
            self.end = True
            return []

        if random() < s["add-probability"]:
            return []

        cnt = randint(s["add-cnt-min"], s["add-cnt-max"])
        if cnt == 0:
            return []

        new = self.create_tasks(cnt)
        return new

    def new_tasks_waves(self, time, _):
        waves = self.conf["spawn"]['waves']

        new = []
        for wave in waves:
            if wave["time"] == time:
                new = self.create_tasks(wave["cnt"])
                break

        if time >= self.conf["spawn"]["last_wave"]:
            self.end = True

        return new

    def is_empty(self):
        return self.end and len(self.tasks) == 0

    def task_init_time(self, task):
        return self.additions[task.id]

    def pull_task(self, task):
        task.choose_length(self.conf["timings"])
        self.tasks.remove(task)

    def create_tasks(self, n):
        p = self.conf['params']
        # p = p if type(p) == list else [p]

        q_sum = 0
        for t in p:
            q_sum += t.get('quantity', 1)

        new_tasks = []
        for i in range(n):
            q_rnd = randrange(q_sum)

            i = 0
            q = p[0].get('quantity', 1)
            while q < q_rnd:
                i += 1
                q += p[i].get('quantity', 1)

            typ = p[i]

            task = Task(
                typ['memory'] if isinstance(typ['memory'], int) else randint(typ['memory'][0], typ['memory'][1]),
                typ['disk'] if isinstance(typ['disk'], int) else randint(typ['disk'][0], typ['disk'][1]),
                typ['cpu'] if isinstance(typ['cpu'], int) else randint(typ['cpu'][0], typ['cpu'][1]),
                typ['time'] if isinstance(typ['time'], int) else randint(typ['time'][0], typ['time'][1]),
            )
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

        if self.paid_time <= 0:
            self.reserved = False
            return True
        else:
            return False

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
    def __init__(self, cnf):
        self.machines = []
        self.time = 0
        self.hidden_price = 0
        self.idle = True

        self.conf = cnf
        self.credit_period = self.conf.get('credit_period', 1)

        machines_own = self.conf.get("own", [])
        for m_info in machines_own:
            fm = Machine(m_info["memory"], m_info["disk"], m_info["cpu"])
            fm.fixed = True
            self.add_machine(fm)

        self.remote_conf = self.conf.get("remote", {})
        self.limit = self.remote_conf.get("limit", 0)
        initial_cnt = self.remote_conf.get("initial", 10)
        self.create_machines(initial_cnt)

    def add_machine(self, machine):
        machine.credit_period = self.credit_period
        self.machines.append(machine)

    def create_machines(self, n):
        p = self.remote_conf['params']
        # p = p if type(p) == list else [p]

        q_sum = 0
        for t in p:
            q_sum += t.get('quantity', 1)

        new_machines = []
        for i in range(n):
            q_rnd = randrange(q_sum)

            i = 0
            q = p[0].get('quantity', 1)
            while q < q_rnd:
                i += 1
                q += p[i].get('quantity', 1)

            typ = p[i]

            machine = Machine(
                typ['memory'] if isinstance(typ['memory'], int) else randint(typ['memory'][0], typ['memory'][1]),
                typ['disk'] if isinstance(typ['disk'], int) else randint(typ['disk'][0], typ['disk'][1]),
                typ['cpu'] if isinstance(typ['cpu'], int) else randint(typ['cpu'][0], typ['cpu'][1]),
            )
            self.add_machine(machine)
            new_machines.append(machine)
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
                self.hidden_price += machine.price
                deleted_machines.append(machine)

        return deleted_machines

    def check_new_machines(self):
        if self.limit != 0 and len(self.machines) >= self.limit:
            return []

        new_machines = []

        if random() >= 0.98:
            new_machines += self.create_machines(1)

        return new_machines

    def check_task(self, task):
        for m in self.machines:
            if m.check_task(task):
                return True
        return False

    def get_tasks_cnt(self):
        return sum((len(machine.workload) for machine in self.machines))

    def get_price(self):
        p = self.hidden_price
        for machine in self.machines:
            p += machine.price
        return p

    def go(self):
        exited = False
        for machine in self.machines:
            exited = machine.go() or exited
        self.idle = all(machine.idle for machine in self.machines)
        self.time += 1
        return exited


class Scheduler:
    def __init__(self, tp, cl, gen):
        self.tp = tp
        self.cluster = cl

        self.logger = TaskLogger()
        self.stats = {
            'time': 0,
            'machines-max': 0,
            'machine-time': 0,
            'machine-time-remote': 0,
            'downtime': 0,
            'downtime-remote': 0,
            'tasks-time': 0,
            'tasks': 0,
            'tasks-on-machine-max': 0,
            'occupancy-sum': 0,
            'waiting': 0,
            'waiting-max': 0,
        }
        self.stats_enabled = True

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
                self.tp.pull_task(next_task)
                self.generator.remove_task(next_task)

                wait_time = self.cluster.time - self.tp.task_init_time(next_task)
                self.stats['waiting'] += wait_time
                self.stats['tasks'] += 1
                if wait_time > self.stats['waiting-max']:
                    self.stats['waiting-max'] = wait_time

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

        if self.tp.is_empty() and self.stats_enabled:
            self.stats_enabled = False
            self.logger.add_mark(self.cluster.time)

    def update_stats(self):

        self.stats['time'] += 1

        if len(self.cluster.machines) > self.stats['machines-max']:
            self.stats['machines-max'] = len(self.cluster.machines)

        for machine in self.cluster.machines:
            if machine.reserved:
                self.stats['machine-time'] += 1
                self.stats['occupancy-sum'] += 1 - min(machine.memory_free / machine.memory,
                                                       machine.cpu_free / machine.cpu,
                                                       machine.disk_free / machine.disk)

                if len(machine.workload) > self.stats['tasks-on-machine-max']:
                    self.stats['tasks-on-machine-max'] = len(machine.workload)

                if not machine.fixed:
                    self.stats['machine-time-remote'] += 1

                if machine.idle and len(machine.workload) == 0:
                    self.stats['downtime'] += 1
                    if not machine.fixed:
                        self.stats['downtime-remote'] += 1
                else:
                    self.stats['tasks-time'] += len(machine.workload)

    def loop(self):
        ready = True
        while not self.tp.is_empty() or not cluster.idle:

            if ready:
                write_log(1, "Time: %8d; Tasks cnt: %4d; Pending: %4d" %
                          (self.cluster.time, cluster.get_tasks_cnt(), len(self.tp.tasks)), one_line=True)
                self.run_tasks()
                ready = False

            if self.stats_enabled:
                self.update_stats()

            exited = self.cluster.go()
            if exited:
                exited = False
                ready = True
                for machine in self.cluster.machines:
                    for task in machine.done:
                        self.logger.task_done(
                            task_id=task["task"].id,
                            time=self.cluster.time
                        )
                    del machine.done[:]

            for machine in self.cluster.machines:
                if machine.idle and machine.reserved and not machine.fixed:
                    if machine.id not in self.plan.chains or len(self.plan.chains[machine.id]) == 0:
                        if machine.free():
                            self.logger.machine_idle(
                                machine_id=machine.id,
                                time=self.cluster.time
                            )

            deleted_machines = self.cluster.check_deleted_machines()
            new_machines = self.cluster.check_new_machines()
            new_tasks = self.tp.check_new_tasks(self.cluster.time)

            if new_tasks or new_machines:
                ready = True
                self.plan_outdated = True

            if deleted_machines:
                [self.generator.remove_machine(machine) for machine in deleted_machines]
                write_log(2, "Time: %8d; Deleted machine!" % self.cluster.time)

            if new_machines:
                [self.generator.add_machine(machine) for machine in new_machines]
                write_log(2, "Time: %8d; New machine!" % self.cluster.time)

            if new_tasks:
                [self.generator.add_task(task) for task in new_tasks]
                write_log(2, "Time: %8d; New tasks!" % self.cluster.time)

        for machine in self.cluster.machines:
            self.logger.machine_idle(
                machine_id=machine.id,
                time=self.cluster.time
            )

        return self.cluster.time, self.cluster.get_price()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment an image using QR markers')

    parser.add_argument('config', help='Configuration file')

    parser.add_argument('-r', '--repeat', metavar='R', type=int, help='Repeat simulation R times')
    parser.add_argument('-d', '--draw', dest='draw', action='store_true', help='Draw log (ignored when R > 1)')
    parser.add_argument('-l', '--log', dest='log', action='store_true', help='Enable text logging')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-vv', '--verbose', dest='verbose2', action='store_true', help='Too verbose mode')

    args = parser.parse_args()

    # Read conf
    conf = {}
    with open(args.config, "r") as f:
        conf_text = f.read()
        conf = yaml.load(conf_text)

    machines_info = conf.get("machines", {})
    credit_period = machines_info.get("credit_period", 1)

    tasks_info = conf.get("tasks", {})

    repeat = conf.get("repeat", 1)
    verbosity = conf.get("verbosity", 0)
    draw = conf.get("draw", False)
    logging = conf.get("log", False)

    alg_info = conf["algorithm"]
    alg_name = alg_info["name"]
    alg_settings = alg_info.get("settings", {})

    # Apply args
    if args.repeat:
        repeat = args.repeat
    if args.draw:
        draw = args.draw
    if args.log:
        logging = args.log
    verbosity = 2 if args.verbose2 else (1 if args.verbose else verbosity)

    if repeat > 1 and draw is True:
        raise RuntimeError("Cannot draw result when repeating multiple times")

    set_verbosity(verbosity)

    cl_times = []
    prices = []
    costs = []
    times = []

    cost_sum = 0

    stats_list = []

    def f_scalar(t, p): return t**0.5 * p**0.5

    for r in range(repeat):
        task_pool = TasksPool(tasks_info)
        task_pool.check_new_tasks(time=0, initial=True)

        cluster = Cluster(machines_info)

        generator = None
        if alg_name == 'genetic':
            generator = GeneticGenerator(cluster.machines, task_pool.tasks, alg_settings, func=f_scalar)
        elif alg_name == 'simple':
            generator = SimpleGenerator(cluster.machines, task_pool.tasks, alg_settings, func=f_scalar)
        else:
            print("Unknown algorithm name")
            exit(1)

        scheduler = Scheduler(task_pool, cluster, generator)

        start = time.time()
        cl_time, price = scheduler.loop()
        end = time.time()
        cost = f_scalar(cl_time, price)

        cl_times.append(cl_time)
        prices.append(price)
        costs.append(cost)
        times.append(end - start)

        cost_sum += cost

        write_log(0, "#%3d: Cost: %5.2f [temp avg: %5.2f ]" % (r, cost, cost_sum/(r+1)), True)
        write_log(1, "#%3d: Time: %8d; Price: %8d; Time: %5.2f" % (r, cl_time, price, end - start))
        stats = scheduler.stats
        stats_list.append(stats)
        write_log(1, "Статистика:")
        write_log(1, "#%3d: Время: %3d" % (r, stats['time']))
        write_log(1, "#%3d: Простои: %3d: %5.2f" %
                  (r, stats['downtime-remote'], stats['downtime-remote']/stats['machine-time-remote']))
        write_log(1, "#%3d: Число машин: ср: %5.2f, макс: %3d" %
                  (r, stats['machine-time']/stats['time'], stats['machines-max']))
        write_log(1, "#%3d: Число задач на машине: ср: %5.2f, макс: %3d" %
                  (r, stats['tasks-time']/stats['machine-time'], stats['tasks-on-machine-max']))
        write_log(1, "#%3d: Средняя загруженность машин: %5.2f" % (r, stats['occupancy-sum']/stats['machine-time']))
        write_log(1, "#%3d: Среднее время ожидания задачи: %5.2f" % (r, stats['waiting']/stats['tasks']))
        write_log(1, "#%3d: Макс. время ожидания задачи: %3d" % (r, stats['waiting-max']))

        if logging:
            scheduler.logger.dump_yaml('log_' + str(r) + '.txt')
        if draw:
            scheduler.logger.draw_all(
                filename_prefix=path.splitext(path.basename(args.config))[0],
                credit_period=credit_period
            )

    if repeat > 1:
        stats = stats_list.pop(0)
        keys = list(stats.keys())
        for key in keys:
            if key[-3:] == 'max':
                stats[key + '-sum'] = stats[key]
                for s in stats_list:
                    stats[key + '-sum'] += s[key]
                    if s[key] > stats[key]:
                        stats[key] = s[key]
            else:
                for s in stats_list:
                    stats[key] += s[key]

        write_log(0, "---------------------")
        write_log(0, "          Число машин: ср: %5.2f, макс: %4d, макс уср: %6.2f" %
                  (stats['machine-time']/stats['time'], stats['machines-max'], stats['machines-max-sum']/repeat))
        write_log(0, "Число задач на машине: ср: %5.2f, макс: %4d, макс уср: %6.2f" %
                  (stats['tasks-time']/stats['machine-time'], stats['tasks-on-machine-max'], stats['tasks-on-machine-max-sum']/repeat))
        write_log(0, "Время ожидания задачи: ср: %5.2f, макс: %4d, макс уср: %6.2f" %
                  (stats['waiting']/stats['tasks'], stats['waiting-max'], stats['waiting-max-sum']/repeat))
        write_log(0, "Средняя загруженность машин: %2d%%" % (100*stats['occupancy-sum']/stats['machine-time']))
        write_log(0, "Простои: %4.2f%%" % (100*stats['downtime-remote']/stats['machine-time-remote']))

        write_log(0, "---------------------")
        write_log(0, "Mean cluster time: %8d; mean price: %8d; mean cost: %5.2f; mean time: %5.2f" %
                  (mean(cl_times), mean(prices), mean(costs), mean(times)))
        write_log(0, "         variance: %8d;   variance: %8d;  variance: %5.2f;  variance: %5.2f" %
                  (sqrt(variance(cl_times)), sqrt(variance(prices)), sqrt(variance(costs)), sqrt(variance(times))))
