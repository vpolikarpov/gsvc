from sys import stdout

def with_next(iterable):
    iterator = iter(iterable)
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (current_item, next_item)
        current_item = next_item
    yield (current_item, None)


# Logging
verbosity = 0
same_line = False

def write_log(level, text, one_line=False):
    if level > verbosity:
        return

    global same_line
    if not one_line and not same_line:
        stdout.write(text + "\n")
    elif one_line and same_line:
        stdout.write("\r" + text)
    elif one_line:
        stdout.write(text)
    elif same_line:
        stdout.write("\n" + text + "\n")
    stdout.flush()

    same_line = one_line


def set_verbosity(v):
    global verbosity
    verbosity = v
