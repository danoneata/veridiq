from pathlib import Path


def image(path, relative_to=None, options=[]):
    path = Path(path)
    if relative_to is not None:
        path = path.relative_to(relative_to)
    if options:
        options_str = "[" + ",".join(options) + "]"
    else:
        options_str = ""
    return r"\includegraphics" + options_str + "{" + str(path) + "}"


def tabular(table):
    return " \\\\ \n".join(" & ".join(row) for row in table)


def macro(name, *args):
    f = lambda x: "{" + str(x) + "}"
    args_str = "".join(map(f, args))
    return "\\" + name + args_str


def multicol(content, n_cols, align):
    return r"\multicolumn{" + str(n_cols) + "}{" + align + "}{" + content + "}"
