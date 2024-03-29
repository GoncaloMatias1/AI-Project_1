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

```python
def hill_climbing_schedule_landings(airplane_stream):
    for airplane in airplane_stream:
        airplane.is_urgent = (airplane.fuel_level_final < airplane.emergency_fuel or
                              airplane.remaining_flying_time < airplane.expected_landing_time)

    landing_schedule_df = schedule_landings(airplane_stream)

    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []

    while True:
        neighbors = get_successors(landing_schedule_df, airplane_stream)

        next_state_df = landing_schedule_df
        next_score = current_score

        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < next_score:
                next_state_df = neighbor_df
                next_score = score

        if next_score == current_score:
            break

        landing_schedule_df = next_state_df
        current_score = next_score

    return landing_schedule_df, scores
```

### Simulated Annealing
Inspired by a metal cooling process, this method searches for the best landing schedule by sometimes allowing worse schedules in the short term to avoid getting stuck in less optimal solutions. It's great for finding a good schedule even when the solution space is complex and full of traps.

```python
def simulated_annealing_schedule_landings(airplane_stream):
    def evaluate_adjusted_landing_schedule(schedule_df):
        landing_schedule_df = schedule_df.copy()
        total_score = 0
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < row['Actual Landing Time']
                difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
                urgency_penalty = 100 if is_urgent else 0
                score = 1000 - difference - urgency_penalty
                landing_schedule_df.at[index, 'Score'] = score
        total_score = landing_schedule_df['Score'].sum()
        return total_score

    current_schedule = schedule_landings(airplane_stream)
    current_score = evaluate_adjusted_landing_schedule(current_schedule)
    T = 1.0  # Initial high temperature
    T_min = 0.001  # Minimum temperature
    alpha = 0.9  # Cooling rate

    while T > T_min:
        i = 0
        while i <= 100:
            new_schedule = current_schedule.copy()
            new_schedule = get_successors(new_schedule, airplane_stream)[0]
            new_score = evaluate_adjusted_landing_schedule(new_schedule)
            delta = new_score - current_score
            if delta < 0 or math.exp(-delta / T) > random.uniform(0, 1):
                current_schedule = new_schedule
                current_score = new_score
            i += 1
        T = T * alpha

    return current_schedule, current_score
```

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