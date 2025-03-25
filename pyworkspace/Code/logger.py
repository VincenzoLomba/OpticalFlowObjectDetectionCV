
from miscellaneous import getTime

activeTask = None

def setActive(task):
    newLine()
    global activeTask
    activeTask = task.upper()
    log("Start of the execution of " + activeTask + "!")

def newLine():
    if activeTask: log("")
    else: print("")

def log(word):
    print("[" + activeTask + "][" + str(getTime()) + "] " + word)

def error(word):
    log(word)
    raise Exception(word)
