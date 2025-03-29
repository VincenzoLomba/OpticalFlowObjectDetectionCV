
from datetime import datetime
import matplotlib.pyplot as plt
dateFormat = "%H:%M:%S" # "%Y-%m-%d_%H-%M-%S"

def getTimeDifferenceAsString(endingTime, startingTime):
    timeDifference = endingTime - startingTime
    hours, remainder = divmod(timeDifference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"

def getTime(): return datetime.now().strftime(dateFormat)

def plot(vector, title):
    plt.figure(figsize=(10, 5))
    # plt.plot(vector, marker='o', linestyle='-', color='b')
    plt.plot(vector, linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('time')
    plt.grid(True)
    plt.show()

def histogram(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Values Distribution')
    plt.xlabel('Values')
    plt.ylabel('Occurrences')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
