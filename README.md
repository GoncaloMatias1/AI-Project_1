# AI-Project - Airport Landing Scheduler

The Airport Landing Scheduler is a sophisticated algorithm designed to manage the continuous stream of airplanes approaching an airport equipped with three landing strips. Its core objective is to ensure the safety and efficiency of airplane landings, taking into account each plane's fuel level, consumption rate and expected arrival time.

## Objective

The algorithm aims to achieve two main goals:

- Safety First: Guarantee that all airplanes land with at least one hour of fuel reserve to avoid fuel-related incidents.

- Optimal Scheduling: Ensure planes land as close to their expected landing times as possible, prioritizing planes that cannot safely wait due to low fuel levels.


## Constraints

- Airplanes consume fuel at a constant rate.
Landing schedules should adhere to expected times to maintain airport operational efficiency.

- The risk of crashes must be minimized while optimizing landing efficiency.

- Each landing strip is occupied for 3 minutes post-landing, limiting the airport to a maximum of 60 landings per hour.

## Critical Failures

The algorithm fails if it results in any plane crashing during the landing process.

## Objective Function

Minimize the risk of crashes while optimizing fuel efficiency and adherence to the expected landing times.

## Input

A continuous stream of airplanes characterized by their fuel levels and expected landing times.

### Generating Airplane Stream

```python
import random
import pandas as pd

class Airplane:
    def __init__(self):
        self.arriving_fuel_level = random.uniform(1000, 5000)  # Fuel level in liters
        self.fuel_consumption_rate = random.uniform(5, 20)  # Liters per minute
        self.expected_landing_time = random.uniform(10, 120)  # Minutes from now

def generate_airplane_stream(num_airplanes):
    return [Airplane() for _ in range(num_airplanes)]


airplane_stream = generate_airplane_stream(50)
df = pd.DataFrame([(i, airplane.arriving_fuel_level, airplane.fuel_consumption_rate, airplane.expected_landing_time)
                   for i, airplane in enumerate(airplane_stream, start=1)],
                  columns=["Airplane", "Fuel", "Consumption Rate", "Expected Landing Time"])

```
## Output

The algorithm produces a landing schedule that prioritizes safety and efficiency, considering fuel levels and expected landing times.

## Evaluation Criteria

- Landings per Hour: The ability to handle up to 60 landings per hour.
- Time Scale: Capability to manage expected landings up to 24 hours in advance.
- Crashes: The algorithm aims for zero crashes.

## Optimization Algorithms

This project utilizes several optimization algorithms, including Hill Climbing, Simulated Annealing, Tabu Search and Genetic Algorithms, to find the most efficient landing schedules under various constraints.

### Hill Climbing
This algorithm improves landing schedules by making small, beneficial changes until no further improvements are found. It's like climbing a hill step by step to reach the top, where the top represents the best possible schedule.

### Simulated Annealing
Inspired by a metal cooling process, this method searches for the best landing schedule by sometimes allowing worse schedules in the short term to avoid getting stuck in less optimal solutions. It's great for finding a good schedule even when the solution space is complex and full of traps.

### Tabu Search
This approach keeps track of previously explored schedules to avoid revisiting them. By remembering where it's already been, it efficiently finds the best landing schedule without wasting time on bad options.

### Genetic Algorithms
Mimicking natural evolution, this method generates a variety of landing schedules and iteratively refines them through processes akin to natural selection and genetic mutation. It's effective for exploring a wide range of possible schedules and evolving them into the best solution over time.

## Getting Started
To use this project, clone the repository, and ensure you have Python and the required libraries installed. Then, generate an airplane stream and pass it to the optimization algorithm of your choice to receive the optimized landing schedule.

```bash
git clone https://github.com/GoncaloMatias1/AI-Project-23-24.git
cd < to the downloaded repository >
python main.py

```

## Trabalho Realizado por:

- Tiago Simões - up202108857
- Gonçalo Matias - up202108703
- Fernando Afonso - up202108686

### FEUP - Inteligência Artificial