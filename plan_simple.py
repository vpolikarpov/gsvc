from plan import WorkPlan, PlanGenerator
from random import shuffle


class SimpleGenerator (PlanGenerator):

    def get_plan(self):
        selected_plan = None
        min_cost = -1

        tries = self.settings.get('tries', 1)

        for i in range(tries):
            shuffle(self.machines)
            plan = self.make_one_plan()
            cost = plan.evaluate(self.machines, self.tasks)
            if selected_plan is None or cost < min_cost:
                min_cost = cost
                selected_plan = plan
        return selected_plan

    def make_one_plan(self):
        machines = [machine.clone() for machine in self.machines_external]
        machines = [m for m in machines if m.fixed] + [m for m in machines if not m.fixed]
        tasks = self.tasks[:]

        plan = WorkPlan()

        if len(tasks) == 0:
            return plan

        while True:
            used = []
            for task in tasks:
                for machine in machines:
                    if machine.check_task(task):
                        machine.reserve()
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
