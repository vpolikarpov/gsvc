from PIL import Image, ImageDraw
import yaml
from math import ceil


class TaskLogger:
    def __init__(self):
        self.log_tasks = []
        self.log_machines = []
        self.path = './log/'

    def task_started(self, task_id, machine_id, time, resources):
        self.log_tasks.append({
            "task_id": task_id,
            "machine_id": machine_id,
            "start": time,
            "end": -1,
            "resources": resources,
        })

    def task_done(self, task_id, time):
        for log_item in self.log_tasks:
            if log_item["task_id"] == task_id and log_item["end"] == -1:
                log_item["end"] = time

    def machine_works(self, machine_id, time, resources):
        for log_item in self.log_machines:
            if log_item["machine_id"] != machine_id:
                continue

            log_item["periods"].append((time, -1))
            return

        self.log_machines.append({
            "machine_id": machine_id,
            "periods": [(time, -1)],
            "resources": resources,
        })

    def machine_idle(self, machine_id, time):
        for log_item in self.log_machines:
            if log_item["machine_id"] == machine_id and log_item["periods"][-1][1] == -1:
                log_item["periods"][-1] = (log_item["periods"][-1][0], time)

    def dump_yaml(self, filename=None):
        text = yaml.dump({"tasks": self.log_tasks, "machines": self.log_machines})
        if filename:
            fo = open(self.path + filename, 'w')
            fo.write(text)
            fo.close()
        else:
            print(text)

    def draw_resource(self, resource, filename=None, credit_period=1):
        def my_append(lst, tpl):
            for i in range(len(lst)):
                if lst[i][0] == tpl[0]:
                    lst[i] = (tpl[0], lst[i][1] + tpl[1], lst[i][2] + tpl[2])
                    return
            lst.append(tpl)

        filename = self.path + (filename or 'log_' + resource + '.png')

        time = max((e['end'] for e in self.log_tasks))

        total_res = sum(item["resources"][resource] for item in self.log_machines)
        img_w = time * 10
        img_h = total_res * 10 + 10 * len(self.log_machines)

        img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        y_start = 5
        for m_log_item in self.log_machines:
            y_end = y_start + m_log_item["resources"][resource] * 10
            draw.line([(0, y_start), (img_w - 1, y_start)], fill=(0, 0, 0))
            draw.line([(0, y_end + 1), (img_w - 1, y_end + 1)], fill=(0, 0, 0))

            for period in m_log_item["periods"]:
                period_end = period[0] + ceil((period[1] - period[0]) / credit_period) * credit_period
                draw.rectangle([period[0] * 10, y_start + 1, period_end * 10, y_end], fill=(222, 222, 255))

            draw.text((2, y_start + 2), "#" + str(m_log_item["machine_id"]), fill=(255, 0, 0))

            start_val = 0
            dif = []
            for ent in self.log_tasks:
                if ent["machine_id"] != m_log_item["machine_id"]:
                    continue

                my_append(dif, (ent["start"], ent["resources"][resource], 1))
                my_append(dif, (ent["end"], -ent["resources"][resource], -1))

                dif.sort(key=lambda t: t[0])

            last_x = -1
            last_y = y_end - start_val * 10
            cnt = 0
            pts = [(0, last_y)]
            for d in dif:
                x = d[0] * 10
                y = last_y - d[1] * 10
                cnt += d[2]
                if x == last_x:
                    del pts[-1]
                else:
                    pts.append((x, last_y))
                pts.append((x, y))
                draw.text((x + 2, y + 2), str(cnt), fill=(255, 0, 0))

                last_y = y
                last_x = x

            draw.line(pts, fill=(255, 0, 0))

            y_start = y_end + 10

        img.save(filename, "PNG")

    def draw_all(self, credit_period=1):
        self.draw_resource('memory', credit_period=credit_period)
        self.draw_resource('disk', credit_period=credit_period)
        self.draw_resource('cpu', credit_period=credit_period)
