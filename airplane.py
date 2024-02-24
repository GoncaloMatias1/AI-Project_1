class Airplane:
    """
    Represents an airplane with its fuel levels, consumption rate, and landing information.
    """

    def __init__(self, airplane_id):
        """
        Initializes an Airplane object with random attributes.

        Args:
            airplane_id (int): The unique identifier of the airplane.
        """
        self.airplane_id = airplane_id
        self.fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_arrival_time = random.uniform(10, 120)
        self.priority = 0

    def calculate_priority(self):
        """
        Calculates the priority of the airplane based on its fuel level and expected arrival time.
        """
        remaining_fuel_at_arrival = self.fuel_level - (self.fuel_consumption_rate * self.expected_arrival_time)
        safety_threshold = self.fuel_consumption_rate * 60
        self.priority = safety_threshold - remaining_fuel_at_arrival
        self.emergency = self.priority < 0
