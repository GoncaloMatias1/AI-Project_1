import pandas as pd

def get_input(prompt, type_=None, min_=None, max_=None, range_=None):
    """Generic function for getting validated user input."""
    while True:
        try:
            value = type_(input(prompt))
            if ((min_ is not None and value < min_) or
                (max_ is not None and value > max_) or
                (range_ is not None and value not in range_)):
                raise ValueError
        except ValueError:
            print("Invalid input. ", end="")
            if type_ is int:
                print("Please enter an integer. ", end="")
            elif type_ is float:
                print("Please enter a number. ", end="")
            if min_ is not None and max_ is not None:
                print(f"Enter a value between {min_} and {max_}.")
            elif min_ is not None:
                print(f"Enter a value greater than {min_}.")
            elif max_ is not None:
                print(f"Enter a value less than {max_}.")
            if range_ is not None:
                print(f"Enter one of the following values: {', '.join(map(str, range_[:-1]))}, or {range_[-1]}.")
        else:
            return value

def validate_airplane_data(airplane):
    """Validate the airplane data for realistic simulation parameters."""
    if not 1000 <= airplane.fuel_level <= 5000:
        raise ValueError("Fuel level must be between 1000 and 5000 liters.")
    if not 5 <= airplane.fuel_consumption_rate <= 20:
        raise ValueError("Fuel consumption rate must be between 5 and 20 liters per minute.")
    if not 10 <= airplane.expected_arrival_time <= 120:
        raise ValueError("Expected landing time must be between 10 and 120 minutes.")

def calculate_statistics(landing_schedule):
    """Calculate statistics from the landing schedule."""
    df = pd.DataFrame(landing_schedule, columns=['Airplane ID', 'Landing Time'])
    average_wait_time = df['Landing Time'].mean()
    return {
        'average_wait_time': average_wait_time,
        'max_wait_time': df['Landing Time'].max(),
        'min_wait_time': df['Landing Time'].min()
    }

def save_simulation_results(results, filename):
    """Save the simulation results to a file."""
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

def load_simulation_results(filename):
    """Load simulation results from a file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]
