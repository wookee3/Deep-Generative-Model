import os


def get_project_root(project_name, os_name="windows"):
    sep = "\\" if os_name == "windows" else "/"
    dirs = os.getcwd().split(sep)
    idx = dirs.index(project_name)
    return sep.join(dirs[:(idx+1)])
