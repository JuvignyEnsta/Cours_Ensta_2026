import matplotlib.pyplot as plt
import pandas as pd
import sys

baseline_time = float(sys.argv[1])

# Load your data
df = pd.read_csv('results.csv')

# Calculate Speedup: Time(1 core) / Time(N cores)
t1 = df[df['cores'] == 1]['time'].values[0]
df['speedup'] = t1 / df['time']

plt.figure(figsize=(12, 5))

# Plot 1: Execution Time
plt.subplot(1, 2, 1)
plt.plot(df['cores'], df['time'], marker='o', color='red')
plt.title('Execution Time vs. Cores')
plt.axhline(y=baseline_time, color='r', linestyle='--', label='mandelbrot_vec')
plt.xlabel('Number of Cores')
plt.ylabel('Time (s)')
plt.grid(True)

# Plot 2: Speedup (The efficiency graph)
plt.subplot(1, 2, 2)
plt.plot(df['cores'], df['speedup'], marker='s', color='blue', label='Actual Speedup')
plt.plot(df['cores'], df['cores'], '--', color='gray', label='Ideal (Linear)')
plt.title('Speedup (Amdahl\'s Law)')
plt.xlabel('Number of Cores')
plt.ylabel('Speedup Factor')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()