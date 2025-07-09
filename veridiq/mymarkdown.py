def ul(xs):
    return "\n".join("- {}".format(x) for x in xs)


def code(x):
    return "```\n{}\n```".format(x)