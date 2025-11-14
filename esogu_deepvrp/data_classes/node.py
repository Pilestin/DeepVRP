"""
Node, Depot ve Customer sınıfları.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """Base class for all location nodes."""
    no: str
    name: str
    latitude: float
    longitude: float
    x: float
    y: float
    
    def get_coordinates(self):
        """Returns (x, y) coordinates."""
        return (self.x, self.y)
    
    def get_gps(self):
        """Returns (lat, lon) GPS coordinates."""
        return (self.latitude, self.longitude)


@dataclass
class Depot(Node):
    """Depot/Charging station."""
    node_type: str = "Depot"
    
    def __repr__(self):
        return f"Depot(name={self.name}, no={self.no}, pos=({self.x:.2f}, {self.y:.2f}))"


@dataclass
class Customer(Node):
    """Customer delivery/pickup location."""
    node_type: str
    weight: int
    quantity: int
    ready_time: int
    service_time: int
    due_date: int
    
    def get_time_window(self):
        """Returns (ready_time, due_date) tuple."""
        return (self.ready_time, self.due_date)
    
    def is_delivery(self):
        """Check if this is a delivery node."""
        return self.node_type == "Delivery"
    
    def is_pickup(self):
        """Check if this is a pickup node."""
        return self.node_type == "Pickup"
    
    def __repr__(self):
        return (f"Customer(name={self.name}, type={self.node_type}, "
                f"demand={self.weight}, tw=[{self.ready_time}, {self.due_date}])")
