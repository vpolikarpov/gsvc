from PIL import Image, ImageDraw, ImageFont
import yaml
from math import ceil
from copy import deepcopy


class TaskLogger:
    def __init__(self):
        self.log_tasks = []
        self.log_machines = []
        self.path = './log/'
        self.marks = []

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

    def add_mark(self, time, text="", color=(255, 255, 0)):
        self.marks.append({
            "text": text,
            "color": color,
            "time": time,
        })

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

        mn_fnt = ImageFont.truetype('arial.ttf', size=50)

        y_start = 5
        for m_log_item in self.log_machines:
            y_end = y_start + m_log_item["resources"][resource] * 10
            draw.line([(0, y_start), (img_w - 1, y_start)], fill=(0, 0, 0))
            draw.line([(0, y_end + 1), (img_w - 1, y_end + 1)], fill=(0, 0, 0))

            for period in m_log_item["periods"]:
                period_end = period[0] + ceil((period[1] - period[0]) / credit_period) * credit_period
                draw.rectangle([period[0] * 10, y_start + 1, period_end * 10, y_end], fill=(235, 235, 255))

            # TASKS
            events = []
            for task in self.log_tasks:
                if task["machine_id"] != m_log_item["machine_id"]:
                    continue

                events.append({
                    'time': task["start"],
                    'id': task["task_id"],
                    'res': task["resources"][resource],
                    'type': 1})
                events.append({
                    'time': task["end"],
                    'id': task["task_id"],
                    'res': task["resources"][resource],
                    'type': -1})

            events.sort(key=lambda t: t['time'])

            colors = [
                (255, 204, 102),
                (255, 102, 51),
                (255, 204, 204),
                (255, 102, 153),
                (204, 102, 255),
                (204, 204, 255),
                (102, 153, 255),
                (153, 255, 204),
                (102, 255, 153),
                (102, 255, 0),
            ]
            color_id = 0

            bounds = []

            if len(events) > 0:
                active = []
                time_prev = 0
                i = 0
                while True:
                    time_new = events[i]['time']
                    if time_new != 0:
                        x1 = time_prev * 10
                        x2 = time_new * 10
                        y1 = y_end

                        for line in active:
                            y2 = y_end - line["pos"] * 10
                            draw.rectangle(((x1, y1), (x2 - 1, y2)), fill=colors[line["color"]])
                            y1 = y2

                    active_new = deepcopy(active)
                    deleted = []

                    cnt = len(events)
                    while i < cnt and events[i]['time'] == time_new:
                        ev = events[i]
                        if ev['type'] == 1:
                            place = 0
                            while place < len(active_new):
                                h = active_new[place]['pos'] - (active_new[place-1]['pos'] if place > 0 else 0)
                                if h > ev['res']:
                                    break
                                place += 1

                            color_id = (color_id + 1) % 10

                            pos = active[place-1]['pos'] if len(active) and place > 0 else 0
                            active.insert(place, {'id': ev["id"], 'pos': pos})

                            pos_new = active_new[place-1]['pos'] if len(active_new) and place > 0 else 0
                            active_new.insert(place, {'id': ev["id"], 'pos': pos_new, 'color': color_id})

                            for j in range(place, len(active_new)):
                                active_new[j]["pos"] += ev['res']

                        else:
                            change = 0
                            id = 0
                            for line in active_new:
                                if line["id"] == ev['id']:
                                    change += ev['res']
                                line["pos"] -= change

                            deleted.append(ev['id'])

                        i += 1

                    if i == cnt:
                        break

                    y1_prev = y_end
                    y2_prev = y_end
                    for j in range(len(active)):
                        y1 = y_end - active[j]["pos"] * 10
                        y2 = y_end - active_new[j]["pos"] * 10
                        x = time_new * 10

                        if "color" in active[j] and active[j]["id"] not in deleted and y1 != y2:
                            bounds.append([((x, min(y1, y2)), (x, max(y1_prev, y2_prev))),
                                           colors[active_new[j]["color"]], 3])

                        y1_prev = y1
                        y2_prev = y2

                    time_prev = time_new
                    active = [ent for ent in active_new if ent["id"] not in deleted]

            for bound in bounds:
                draw.line(*bound)

            # ENVELOPE
            start_val = 0
            dif = []
            for line in self.log_tasks:
                if line["machine_id"] != m_log_item["machine_id"]:
                    continue

                my_append(dif, (line["start"], line["resources"][resource], 1))
                my_append(dif, (line["end"], -line["resources"][resource], -1))

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

            draw.line(pts, fill=(255, 0, 0), width=3)

            draw.text((10, y_end - 60), "#" + str(m_log_item["machine_id"]), fill=(0, 0, 0), font=mn_fnt)

            y_start = y_end + 10

        for mark in self.marks:
            pos = mark['time'] * 10 + 1
            draw.line([(pos, 0), (pos, img_h - 1)], mark['color'])
            draw.text((pos + 2, 2), mark['text'], mark['color'])

        img.save(filename, "PNG")

    def draw_all(self, filename_prefix='log', credit_period=1):
        name = '%s_%%s.png' % filename_prefix
        self.draw_resource('memory', filename=(name % 'memory'), credit_period=credit_period)
        self.draw_resource('disk', filename=(name % 'disk'), credit_period=credit_period)
        self.draw_resource('cpu', filename=(name % 'cpu'), credit_period=credit_period)
