import random
import pandas as pd
import numpy as np


class Airplane:
    def __init__(self, id, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
        self.id = id
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        self.fuel_level = random.uniform(min_fuel, max_fuel)

        self.emergency_fuel = (self.fuel_consumption_rate * 60)
        self.fuel_level = max(self.fuel_level, self.emergency_fuel)
        self.fuel_level_final = self.fuel_level - self.emergency_fuel
        self.remaining_flying_time = self.fuel_level_final / self.fuel_consumption_rate

        self.is_urgent = self.fuel_level_final < self.emergency_fuel or self.remaining_flying_time < 1


def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    sorted_airplanes = sorted(airplane_stream, key=lambda x: x.expected_landing_time)
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]  
    landing_strip_index = 0

    for airplane in sorted_airplanes:
        if airplane.fuel_level_final == 0:
            airplane.fuel_level_final = airplane.fuel_level

        chosen_strip = landing_strip_index % 3
        next_available_time_with_gap = landing_strip_availability[chosen_strip] + 3/60
        is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
        actual_landing_time = max(airplane.expected_landing_time, next_available_time_with_gap)

        if is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time 

        if airplane.fuel_level_final == airplane.emergency_fuel:
            actual_landing_time = max(actual_landing_time, airplane.expected_landing_time)

        landing_strip_availability[chosen_strip] = actual_landing_time + 3
        landing_schedule.append((airplane.id, actual_landing_time, is_urgent, chosen_strip + 1))

        landing_strip_index += 1

    return pd.DataFrame(landing_schedule, columns=["Airplane ID", "Actual Landing Time", "Urgent", "Landing Strip"])

#hill climbing
def evaluate_landing_schedule(landing_schedule_df, airplane_stream):
    for index, row in landing_schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
            urgency_penalty = 100 if airplane.is_urgent else 0
            score = 1000 - difference - urgency_penalty
            landing_schedule_df.at[index, 'Score'] = score

    total_score = landing_schedule_df['Score'].sum()
    return total_score

#hill climbing
def get_successors(landing_schedule_df, airplane_stream): #esta funcao faz parte do hill climbing, o que faz é gerar sucessores para o estado atual fazendo pequenas alterações nos tempos de aterragem
    successors = []
    for i in range(len(landing_schedule_df)): #este loop irá iterar sobre todos os aviões de modo a gerar sucessores
        for j in range(i + 1, len(landing_schedule_df)): 
            # esta dataframe é uma cópia da original, de modo a que possamos fazer alterações sem afetar o estado atual
            new_schedule_df = landing_schedule_df.copy()
            # estamos a trocar os tempos de aterragem de dois aviões, de modo a gerar um sucessor
            # o sucessor serve para que possamos comparar o score do estado atual com o score do sucessor
            new_schedule_df.iloc[i], new_schedule_df.iloc[j] = new_schedule_df.iloc[j].copy(), new_schedule_df.iloc[i].copy()
            successors.append(new_schedule_df)
    return successors

def get_tabu_successors(landing_schedule_df, airplane_stream, tabu_list, current_score, num_neighbors=10):
    neighbors = []
    for _ in range(num_neighbors):
        # Choose two random airplanes
        i, j = random.sample(range(len(airplane_stream)), 2)
        # Create a copy of the current landing schedule
        neighbor_df = landing_schedule_df.copy()
        # Switch the two airplanes and their landing strip assignments
        neighbor_df.iloc[i], neighbor_df.iloc[j] = landing_schedule_df.iloc[j].copy(), landing_schedule_df.iloc[i].copy()
        # Convert the neighbor to a string to check if it's in the tabu list
        neighbor_str = neighbor_df.to_string()
        # Calculate the score of the neighbor
        neighbor_score = evaluate_landing_schedule(neighbor_df, airplane_stream)
        # If the neighbor is not in the tabu list, or if it's better than the current solution (aspiration criterion), add it to the list of neighbors
        if neighbor_str not in tabu_list or neighbor_score > current_score:
            neighbors.append(neighbor_df)
    return neighbors

def generate_initial_schedule(airplane_stream):
    shuffled_stream = random.sample(airplane_stream, len(airplane_stream))
    return schedule_landings(shuffled_stream)

def select_parents(population, fitness_scores, num_parents):
    fitness_scores = np.array(fitness_scores)
    probabilities = 1 / (1 + fitness_scores)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(range(len(population)), size=num_parents, replace=False, p=probabilities)
    return [population[i] for i in selected_indices]

def crossover(parents, crossover_rate):
    offspring = []
    for _ in range(len(parents) // 2):
        parent1, parent2 = random.sample(parents, 2)
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, parent1.shape[0] - 2)  # asim evitamos extremos
            child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
            child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    return offspring

def mutate(schedule, mutation_rate, airplane_stream):
    for index in range(len(schedule)):
        if random.random() < mutation_rate:
            replacement_plane = random.choice(airplane_stream)
            replacement_index = schedule[schedule['Airplane ID'] == replacement_plane.id].index[0]
            schedule.at[index, 'Actual Landing Time'], schedule.at[replacement_index, 'Actual Landing Time'] = schedule.at[replacement_index, 'Actual Landing Time'], schedule.at[index, 'Actual Landing Time']
    return schedule

#uma pontuação de 0 indica um evento de aterragem ótimo ou sem penalizações

#As pontuações diferentes de zero sugerem penalizações devidas a desvios das condições óptimas, tais como atrasos ou não resposta adequada à urgência devido a pouco combustível.

#O objetivo do AG é minimizar estas pontuações, procurando obter uma pontuação total de 0 no programa, o que indica que não há penalizações em todos os eventos de aterragem e, por conseguinte, um programa ótimo, tendo em conta as restrições e os objectivos definidos.