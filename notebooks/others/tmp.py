#show current time in 12 hour format
import datetime

now = datetime.datetime.now()
print(now.strftime("%I:%M %p"))

#show current time in 24 hour format
print(now.strftime("%H:%M"))

#show current date
print(now.strftime("%Y-%m-%d"))

