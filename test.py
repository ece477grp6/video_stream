import datetime
start_time = datetime.datetime.now()
current_time = datetime.datetime.now()
a = (current_time-start_time).total_seconds()
print(a)
