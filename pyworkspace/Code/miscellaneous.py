
from datetime import datetime
dateFormat = "%H:%M:%S" # "%Y-%m-%d_%H-%M-%S"

def getTimeDifferenceAsString(endingTime, startingTime):
    timeDifference = endingTime - startingTime
    hours, remainder = divmod(timeDifference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"

def getTime(): return datetime.now().strftime(dateFormat)