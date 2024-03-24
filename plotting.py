import matplotlib.pyplot as plt

def plot_scores(times, scores, algorithm_name, filename='algorithm_performance.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, marker='o')  # Uses times for the x-axis
    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title(f'{algorithm_name} Performance Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

