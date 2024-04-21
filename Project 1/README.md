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
    # Mark urgent airplanes based on their fuel levels and expected landing times.
    for airplane in airplane_stream:
        airplane.is_urgent = (airplane.fuel_level_final < airplane.emergency_fuel or
                                airplane.remaining_flying_time < airplane.expected_landing_time)

    # Generate an initial landing schedule using the schedule_landings function.
    landing_schedule_df = schedule_landings(airplane_stream)

    # Initialize the current score and a list to store the scores of each iteration.
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []

    # Repeat the following steps until no improvement is found.
    while True:
        # Get all neighboring landing schedules from the current schedule.
        neighbors = get_Hill_Tabu_successors(landing_schedule_df, airplane_stream)

        # Assume the next state is the same as the current state and track the highest score.
        next_state_df = landing_schedule_df
        next_score = current_score

        # Iterate over the neighboring landing schedules and find the one with the highest score.
        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score > next_score:
                next_state_df = neighbor_df
                next_score = score

        # If the next score is equal to the current score, indicating no improvement, the search terminates.
        if next_score == current_score:
            break

        # Update the current state and score to the next state and score.
        landing_schedule_df = next_state_df
        current_score = next_score

    # Return the optimized landing schedule and an empty list of scores.
    return landing_schedule_df, scores
```

### Simulated Annealing
Inspired by a metal cooling process, this method searches for the best landing schedule by sometimes allowing worse schedules in the short term to avoid getting stuck in less optimal solutions. It's great for finding a good schedule even when the solution space is complex and full of traps.

```python
def simulated_annealing_schedule_landings(airplane_stream):
    def calculate_score(schedule_df, airplane_stream):
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                time_diff = abs(airplane.expected_landing_time - row['Actual Landing Time'])
                urgency_penalty = 100 if airplane.is_urgent else 0
                score = 1000 - time_diff - urgency_penalty
                schedule_df.at[index, 'Score'] = score
        return schedule_df

    def get_schedule_neighbor(schedule_df):
        neighbor_df = schedule_df.copy()
        i, j = random.sample(range(len(neighbor_df)), 2)
        neighbor_df.iloc[i], neighbor_df.iloc[j] = neighbor_df.iloc[j].copy(), neighbor_df.iloc[i].copy()
        return neighbor_df

    current_schedule = schedule_landings(airplane_stream)
    current_schedule = calculate_score(current_schedule, airplane_stream)
    current_score = current_schedule['Score'].sum()
    best_schedule = current_schedule
    best_score = current_score
    T = 1.0  # Initial high temperature
    T_min = 0.001  # Minimum temperature
    alpha = 0.9  # Cooling rate

    while T > T_min:
        new_schedule = get_schedule_neighbor(current_schedule)
        new_schedule = calculate_score(new_schedule, airplane_stream)
        new_score = new_schedule['Score'].sum()
        if new_score > current_score or math.exp((new_score - current_score) / T) > random.random():
            current_schedule = new_schedule
            current_score = new_score
            if new_score > best_score:
                best_schedule = new_schedule
                best_score = new_score
        T *= alpha  # Cool down

    return best_schedule, best_score
```

### Tabu Search
This approach keeps track of previously explored schedules to avoid revisiting them. By remembering where it's already been, it efficiently finds the best landing schedule without wasting time on bad options.

```python
def tabu_search_schedule_landings(airplane_stream, max_iterations=1000, max_tabu_size=10, patience=3):
    # Mark urgent airplanes based on their fuel levels and expected landing times.
    for airplane in airplane_stream:
        airplane.is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
    
    # Generate an initial landing schedule using the schedule_landings function.
    landing_schedule_df = schedule_landings(airplane_stream)
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []
    tabu_set = set()
    it = 0
    no_improvement_count = 0
    best_score = float('-inf') 

    # Dictionary to store previously evaluated schedules
    evaluated_schedules = {}

    while it < max_iterations and no_improvement_count < patience:
        print(f"Iteration {it}")
        # Get all neighboring landing schedules from the current schedule.
        neighbors = get_Hill_Tabu_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score

        best_solution_df = landing_schedule_df
        best_solution_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)

        # Iterate over the neighboring landing schedules and find the one with the highest score.
        for neighbor_df in neighbors:
            neighbor_hash = hash(neighbor_df.to_string())
            # If we've already evaluated this schedule, retrieve the score from the dictionary
            if neighbor_hash in evaluated_schedules:
                score = evaluated_schedules[neighbor_hash]
            else:
                score = evaluate_landing_schedule(neighbor_df, airplane_stream)
                evaluated_schedules[neighbor_hash] = score

            if score > best_solution_score:
                best_solution_df = neighbor_df
                best_solution_score = score
                # Add only improving solutions to the tabu list.
                if neighbor_hash not in tabu_set:
                    next_state_df = neighbor_df
                    next_score = score
                    tabu_set.add(neighbor_hash) 
                    if len(tabu_set) > max_tabu_size:
                        tabu_set.pop()

        # Aspiration criteria
        if hash(best_solution_df.to_string()) in tabu_set and best_solution_score > best_score:
            next_state_df = best_solution_df
            next_score = best_solution_score
            tabu_set.remove(hash(best_solution_df.to_string()))
            
        # Update the current state and score to the next state and score.
        landing_schedule_df = next_state_df
        current_score = next_score

        # If the best solution score is better than the best score so far, reset the no improvement count.
        # Otherwise, increment the no improvement count.
        if best_solution_score > best_score:
            best_score = best_solution_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Increment the iteration count.
        it += 1
    # Return the best solution found and the list of scores.
    return best_solution_df, scores
```

### Genetic Algorithms
Mimicking natural evolution, this method generates a variety of landing schedules and iteratively refines them through processes akin to natural selection and genetic mutation. It's effective for exploring a wide range of possible schedules and evolving them into the best solution over time.

```python
class GeneticAlgorithmScheduler:
    def __init__(self, airplane_stream, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        self.airplane_stream = airplane_stream
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [self.generate_initial_schedule() for _ in range(self.population_size)]

    def generate_initial_schedule(self):
        shuffled_stream = random.sample(self.airplane_stream, len(self.airplane_stream))
        return schedule_landings(shuffled_stream)

    def calculate_fitness(self, schedule):
        return evaluate_landing_schedule(schedule, self.airplane_stream)

    def selection(self):
        fitness_scores = [self.calculate_fitness(schedule) for schedule in self.population]
        probabilities = 1 / (1 + np.array(fitness_scores))
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(range(len(self.population)), size=self.population_size, replace=False, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, parent1.shape[0] - 2)
            child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
            child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, schedule):
        for index in range(len(schedule)):
            if random.random() < self.mutation_rate:
                replacement_plane = random.choice(self.airplane_stream)
                replacement_index = schedule[schedule['Airplane ID'] == replacement_plane.id].index[0]
                schedule.at[index, 'Actual Landing Time'], schedule.at[replacement_index, 'Actual Landing Time'] = schedule.at[replacement_index, 'Actual Landing Time'], schedule.at[index, 'Actual Landing Time']
        return schedule

    def run(self):
        best_score = float('inf')
        best_schedule = None
        stale_generations = 0

        for generation in range(self.generations):
            new_population = []
            parents = self.selection()

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

            current_best_score = min([self.calculate_fitness(schedule) for schedule in self.population])
            if current_best_score < best_score:
                best_score = current_best_score
                best_schedule = self.population[[self.calculate_fitness(schedule) for schedule in self.population].index(best_score)]
                stale_generations = 0  
            else:
                stale_generations += 1 

            print(f"Generation {generation}: Best Score - {best_score}")

            if stale_generations >= 5:
                print("No improvement over the last 5 generations. Stopping early.")
                break

        return best_schedule, best_score
```


## Getting Started
To use this project, clone the repository, and ensure you have Python and the required libraries installed. Then, generate an airplane stream and pass it to the optimization algorithm of your choice to receive the optimized landing schedule.

```bash
git clone https://github.com/GoncaloMatias1/AI-Project-23-24.git
cd < to the downloaded repository >
python main.py

```

## Project developed by:

- Tiago Simões - up202108857
- Gonçalo Matias - up202108703
- Fernando Afonso - up202108686

### FEUP - Artificial Intelligence
