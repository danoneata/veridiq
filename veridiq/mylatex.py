from pathlib import Path


def image(path, relative_to=None, options=[]):
    path = Path(path)
    if relative_to is not None:
        path = path.relative_to(relative_to)
    if options:
        options_str = "[" + ",".join(options) + "]"
    return r"\includegraphics" + options_str + "{" + str(path) + "}"


def tabular(table):
    return " \\\\ \n".join(" & ".join(row) for row in table)


def macro(name, arg):
    return "\\" + name + "{" + arg + "}"