import time
tq = 0. # Example value for tq
start = time.perf_counter()
for i in range(20):  # Simulate some work
    time.sleep(tq)
actual_sleep = time.perf_counter() - start
print(f"Requested: {tq*1000:.2f}ms, Actual: {actual_sleep*1000:.2f}ms")