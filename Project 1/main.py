import time
import pandas as pd

from simulation import (generate_airplane_stream, hill_climbing_schedule_landings, simulated_annealing_schedule_landings,
                        tabu_search_schedule_landings, GeneticAlgorithmScheduler)


"""
Prompt the user to input various parameters for the simulation.

This function prompts the user to input the number of airplanes, fuel levels, and expected arrival times. It then 
returns these parameters for further use in the simulation.

@return: The number of airplanes, fuel levels, and expected arrival times input by the user.
@rtype: tuple(int, list[float], list[float])
"""

def get_input(prompt, type_=None, min_=None, max_=None, header=None):
    while True:
        if header:
            print("\n" + "-" * 72)
            print("|" + header.center(70) + "|")
            print("-" * 72)
        try:
            value = input("| " + prompt.ljust(68) + "| ")
            print("-" * 72)
            if type_ is not None:
                value = type_(value)
            if (min_ is not None and value < min_) or (max_ is not None and value > max_):
                raise ValueError(f"Value should be between {min_} and {max_}.")
            return value
        except ValueError as e:
            print("|" + f"Invalid input: {e}".center(70) + "|")
            print("-" * 72)


"""
Prompt the user to select an algorithm to optimize the landing schedule.

This function prompts the user to select an algorithm from a list of available algorithms. It then returns the 
selected algorithm for use in the simulation.

@return: The selected algorithm.
@rtype: function
"""

def select_algorithm():
    print("\n" + "-" * 72)
    print("|" + "Airport Landing Scheduler".center(70) + "|")
    print("|" + "Optimization Algorithm Selection".center(70) + "|")
    print("-" * 72)
    print("|" + "1 - Hill Climbing".ljust(69) + "|")
    print("|" + "2 - Simulated Annealing".ljust(69) + "|")
    print("|" + "3 - Tabu Search".ljust(69) + "|")
    print("|" + "4 - Genetic Algorithm".ljust(69) + "|")
    print("-" * 72)
    choice = get_input("Enter your choice (number): ", type_=int, min_=1, max_=4)  # Update max_=4
    return choice

"""
    Main function to run the Airport Landing Scheduler simulation.

    This function first welcomes the user, and then prompts them to input various parameters for the simulation, 
    including the number of airplanes, fuel levels, and expected arrival times.

    It then generates an airplane stream based on these parameters and displays the initial state of the airplane 
    stream in a DataFrame.

    Finally, it prompts the user to select an algorithm to optimize the landing schedule and runs the selected 
    algorithm.

    No parameters are necessary to run this function.

    Returns:
    None
"""

def main():
    print("\n" + "=" * 72)
    print("=" + "Welcome to the Airport Landing Scheduler".center(70) + "=")
    print("=" * 72)
    num_airplanes = get_input("Enter the number of airplanes for the simulation (between 1-1440): ", 
                              type_=int, min_=1, max_=1440, 
                              header="Simulation Setup - Number of Airplanes")
    min_fuel = get_input("Enter the minimum fuel level (in liters, between 1000-5000): ", 
                         type_=float, min_=1000, max_=5000, 
                         header="Simulation Setup - Minimum Fuel Level")
    max_fuel = get_input("Enter the maximum fuel level (in liters, between 1000-5000): ", 
                         type_=float, min_=min_fuel, max_=5000, 
                         header="Simulation Setup - Maximum Fuel Level")
    min_arrival_time = get_input("Enter the minimum expected arrival time (in minutes, between 10-1440): ", 
                                 type_=float, min_=10, max_=1440, 
                                 header="Simulation Setup - Minimum Arrival Time")
    max_arrival_time = get_input("Enter the maximum expected arrival time (in minutes, between 10-1440): ", 
                                 type_=float, min_=min_arrival_time, max_=1440, 
                                 header="Simulation Setup - Maximum Arrival Time")

    airplane_stream = generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df_initial = pd.DataFrame(
        [(airplane.id, airplane.fuel_level, airplane.fuel_level_final, airplane.emergency_fuel,
          airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Initial Fuel", "Final Fuel", "Emergency Fuel Level", "Consumption Rate",
                 "Expected Landing Time"])

    print("\nGenerated Airplane Stream DataFrame:")
    print(df_initial.to_string(index=False))


    algorithm_choice = select_algorithm()

    if algorithm_choice == 1:
        print("Running Hill Climbing algorithm...")
        start_time = time.time()
        landing_schedule_df, scores = hill_climbing_schedule_landings(airplane_stream)
        print("Hill Climbing algorithm finished.")
        print("Final landing schedule:")
        print(landing_schedule_df.to_string(index=False))
        average_score = landing_schedule_df['Score'].mean()
        end_time = time.time()
        print("Execution time: ", end_time - start_time, "seconds")
        print("\nAverage Score: {:.2f}".format(average_score))

    if algorithm_choice == 2:
        print("Running Simulated Annealing algorithm...")
        landing_schedule_df, best_score = simulated_annealing_schedule_landings(airplane_stream)
        print("Simulated Annealing algorithm finished.")
        print("Final landing schedule and score:")
        print(landing_schedule_df.to_string(index=False))
        average_score = landing_schedule_df['Score'].mean()
        print(f"Average Score: {average_score:.2f}")

    elif algorithm_choice == 3:
        max_iterations = get_input("Enter the maximum number of iterations for the Tabu Search algorithm (between 100-1000): ", type_=int, min_=100, max_=1000)
        max_tabu_size = get_input("Enter the maximum size of the tabu list for the Tabu Search algorithm (between 5-15): ", type_=int, min_=5, max_=15)
        patience = get_input("Enter the patience for the Tabu Search algorithm (between 3-10): ", type_=int, min_=3, max_=10)

        print("Running Tabu Search algorithm...")
        start_time = time.time()
        landing_schedule_df, scores = tabu_search_schedule_landings(airplane_stream, max_iterations, max_tabu_size, patience)
        print("Tabu Search algorithm finished.")
        print("Final landing schedule:")
        print(landing_schedule_df.to_string(index=False))
        average_score = landing_schedule_df['Score'].mean()
        end_time = time.time()
        print("Execution time: ", end_time - start_time, "seconds")
        print("\nAverage Score: {:.2f}".format(average_score))

    elif algorithm_choice == 4:
        print("Running Genetic Algorithm...")
        
        genetic_algorithm = GeneticAlgorithmScheduler(airplane_stream)
        
        best_schedule, best_score = genetic_algorithm.run()
        
        print("Genetic Algorithm finished.")
        print(f"Best landing schedule score: {best_score}")
        

        if 'Score' in best_schedule.columns:
            average_score = best_schedule['Score'].mean()
            print("\nAverage Score: {:.2f}".format(average_score))
        else:
            print("\nNo 'Score' column in the schedule to calculate the average.")
    print("Program finished. Exiting...")

        

if __name__ == "__main__":
    main()


