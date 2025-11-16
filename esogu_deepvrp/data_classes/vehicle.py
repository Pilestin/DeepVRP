"""
Vehicle sınıfı - Elektrikli araç özellikleri.
"""

from dataclasses import dataclass


@dataclass
class Vehicle:
    """Electric vehicle for EVRP."""
    vehicle_id: int
    capacity: float = 350.0  # Max load capacity (kg)
    battery_capacity: float = 15600.0  # kWh
    initial_battery: float = 15600.0  # Starting battery level
    
    # Current state
    current_load: float = 0.0
    current_battery: float = 15600.0
    current_location: int = 0  # Node index
    
    def reset(self):
        """Reset vehicle to initial state."""
        self.current_load = 0.0
        self.current_battery = self.initial_battery
        self.current_location = 0
    
    def can_load(self, weight: float) -> bool:
        """Check if vehicle can accommodate additional weight."""
        return self.current_load + weight <= self.capacity
    
    def load(self, weight: float):
        """Add weight to vehicle."""
        if self.can_load(weight):
            self.current_load += weight
            return True
        return False
    
    def unload(self, weight: float):
        """Remove weight from vehicle."""
        self.current_load = max(0, self.current_load - weight)
    
    def consume_battery(self, amount: float):
        """Consume battery energy."""
        self.current_battery = max(0, self.current_battery - amount)
    
    def charge(self, amount: float):
        """Charge battery."""
        self.current_battery = min(self.battery_capacity, self.current_battery + amount)
    
    def __repr__(self):
        return (f"Vehicle(id={self.vehicle_id}, load={self.current_load:.1f}/{self.capacity}, "
                f"battery={self.current_battery:.1f}/{self.battery_capacity})")
