from random import randrange


class WorkPlan:
    def __init__(self, plan=None):
        self.chains = {}
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
            print("Plan Error: Wrong tasks number")
            return False

        # Check tasks duplicates
        tids_plain = [tid for _, chain in self.chains.items() for tid in chain]
        for id_1, tid_1 in enumerate(tids_plain):
            for id_2 in range(id_1 + 1, len(tids_plain)):
                tid_2 = tids_plain[id_2]
                if tid_1 == tid_2:
                    print("Plan Error: Task duplication")
                    return False

        # Check that each task fits to corresponding machine
        for mid, chain in self.chains.items():
            machine = None
            for m in machines:
                if m.id == mid:
                    machine = m

            for tid in chain:
                task = None
                for t in tasks:
                    if t.id == tid:
                        task = t

                if not machine.check_task_clean(task):
                    print("Plan Error: Misfit")
                    return False

        return True

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
