import pandas as pd
from airplane import Airplane  # Import the Airplane class
from landing_schedule import generate_airplane_stream, schedule_landings

def main():
    """
    Runs the simulation, handling user input, generating airplanes, scheduling landings, and displaying results.
    """

    # User input section
    num_airplanes = int(input("Enter number of airplanes: "))
    min_fuel, max_fuel = map(int, input("Enter min and max fuel level (separated by space): ").split())
    min_arrival, max_arrival = map(int, input("Enter min and max arrival time (separated by space): ").split())
    emergency_prob = float(input("Enter probability of emergency landing (0-1): "))

    # Generate the airplane stream based on user input
    airplane_stream = generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival, max_arrival, emergency_prob)

    # Schedule the landings
    landing_schedule = schedule_landings(airplane_stream)

    # Display results
    print("\nSimulation Results:")
    print(f"Number of airplanes: {num_airplanes}")
    print(f"Fuel level range: {min_fuel} - {max_fuel}")
    print(f"Arrival time range: {min_arrival} - {max_arrival} minutes")
    print(f"Emergency probability: {emergency_prob}")

    # Create and display DataFrames (or other output formats)
    df_schedule = pd.DataFrame(landing_schedule, columns=["Airplane ID", "Landing Time (min)"])
    print(df_schedule.to_string())

    # ... (display additional information and statistics)

if __name__ == "__main__":
    main()
