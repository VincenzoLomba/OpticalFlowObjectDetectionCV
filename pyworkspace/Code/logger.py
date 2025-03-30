
from miscellaneous import getTime

activeTask = None

def setActive(task: str):
    newLine()
    global activeTask
    activeTask = task.upper()
    log("Starting the execution of " + activeTask + "!")

def newLine():
    if activeTask: log("")
    else: print("")

def log(word: str):
    print("[" + activeTask + "][" + str(getTime()) + "] " + word)

def error(word: str):
    log(word)
    newLine()
    raise Exception(word)
