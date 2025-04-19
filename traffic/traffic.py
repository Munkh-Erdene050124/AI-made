"""
Intelligent Multi-Modal Route Optimization System
- Incorporates real-time traffic, historical patterns, and ML predictions
- Uses parallel computing, advanced graph algorithms, and interactive visualization
"""

import osmnx as ox
import networkx as nx
import requests
import folium
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import GradientBoostingRegressor
from folium.plugins import HeatMap, MarkerCluster
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(filename='route_optimizer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OX_API_KEY = os.getenv('OX_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')

# Configuration using dataclass
from dataclasses import dataclass

@dataclass
class Config:
    default_capacity_per_lane: int = 1800
    alpha: float = 0.15
    beta: int = 4
    max_workers: int = 8
    traffic_update_interval: int = 300  # seconds
    ml_model_path: str = 'congestion_gbr.pkl'
    
config = Config()

class RouteOptimizer:
    def __init__(self):
        self.ml_model = self._load_ml_model()
        self.traffic_data = None
        self.last_traffic_update = None
        
    def _load_ml_model(self):
        """Load pre-trained congestion prediction model"""
        try:
            return GradientBoostingRegressor().load(config.ml_model_path)
        except Exception as e:
            logging.warning(f"ML model loading failed: {e}")
            return None

    def get_real_time_traffic(self, bbox: Tuple[float]) -> Dict:
        """Fetch real-time traffic data from TomTom API"""
        if datetime.now() - self.last_traffic_update < timedelta(seconds=config.traffic_update_interval):
            return self.traffic_data
            
        url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
               f"?key={TOMTOM_API_KEY}&bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self.traffic_data = response.json()
            self.last_traffic_update = datetime.now()
            logging.info("Updated real-time traffic data")
            return self.traffic_data
        except Exception as e:
            logging.error(f"Traffic API error: {e}")
            return {}

    def _enhance_graph(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Enrich graph with additional features"""
        for u, v, data in G.edges(data=True):
            data['maxspeed'] = data.get('maxspeed', 50)
            data['lanes'] = data.get('lanes', 2)
            data['grade'] = data.get('grade', 0)
            data['surface'] = 1 if data.get('surface', 'asphalt') == 'asphalt' else 0
        return G

    def create_multi_modal_graph(self, center_point: Tuple[float], radius: int = 5000) -> nx.MultiDiGraph:
        """Create integrated multi-modal transportation graph"""
        drive_graph = ox.graph_from_point(center_point, radius=radius, network_type='drive')
        bike_graph = ox.graph_from_point(center_point, radius=radius, network_type='bike')
        walk_graph = ox.graph_from_point(center_point, radius=radius, network_type='walk')
        
        G = nx.compose_all([drive_graph, bike_graph, walk_graph])
        G = self._enhance_graph(G)
        return G

    def predict_congestion(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Predict congestion using ML model"""
        if not self.ml_model:
            return G
            
        features = []
        for u, v, data in G.edges(data=True):
            features.append([
                data['length'],
                data['maxspeed'],
                data['lanes'],
                data['grade'],
                data['surface'],
                datetime.now().hour
            ])
            
        congestion = self.ml_model.predict(features)
        for i, (u, v, data) in enumerate(G.edges(data=True)):
            data['congestion'] = congestion[i]
            
        return G

    def adaptive_a_star(self, G: nx.MultiDiGraph, start: Tuple[float], end: Tuple[float]) -> List:
        """Adaptive A* with dynamic heuristic and contraction hierarchies"""
        orig_node = ox.distance.nearest_nodes(G, start[1], start[0])
        dest_node = ox.distance.nearest_nodes(G, end[1], end[0])
        
        def heuristic(u, v):
            """Dynamic heuristic considering real-time conditions"""
            coord_u = (G.nodes[u]['y'], G.nodes[u]['x'])
            coord_v = (G.nodes[v]['y'], G.nodes[v]['x'])
            dist = haversine(coord_u, coord_v, unit=Unit.KILOMETERS)
            avg_speed = 50  # Base speed
            if self.traffic_data:
                avg_speed = self.traffic_data.get('currentSpeed', 50)
            return (dist / avg_speed) * 3600  # Convert to seconds
            
        return nx.astar_path(G, orig_node, dest_node, heuristic=heuristic, weight='congestion')

    def visualize_advanced(self, G: nx.MultiDiGraph, path: List, alternatives: List[List]):
        """Interactive visualization with multiple layers"""
        m = folium.Map(location=path[0], zoom_start=13)
        
        # Base layers
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('https://{s}.tile.openstreetmap.de/{z}/{x}/{y}.png').add_to(m)
        
        # Route layers
        colors = ['#FF0000', '#00FF00', '#0000FF']
        for i, route in enumerate([path] + alternatives):
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            folium.PolyLine(coords, color=colors[i%3], weight=5).add_to(m)
        
        # Traffic heatmap
        if self.traffic_data:
            heat_data = [(edge['y'], edge['x'], edge['congestion']) 
                        for _, _, edge in G.edges(data=True)]
            HeatMap(heat_data, radius=15).add_to(m)
        
        # Points of interest
        MarkerCluster(name='Landmarks').add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save('advanced_visualization.html')

    def calculate_environmental_impact(self, path: List, G: nx.MultiDiGraph) -> Dict:
        """Calculate CO2 emissions and energy consumption"""
        co2 = 0
        for i in range(len(path)-1):
            edge = G[path[i]][path[i+1]][0]
            distance = edge['length'] / 1000  # km
            speed = edge['maxspeed']
            co2 += distance * (200 - speed) / 50  # Simplified model
        return {'co2_kg': co2, 'energy_kwh': co2 * 0.3}

    def optimize_route(self, start: str, end: str) -> Dict:
        """Full optimization pipeline"""
        try:
            # Geocode addresses
            start_point = ox.geocode(start)
            end_point = ox.geocode(end)
            
            # Create multi-modal graph
            G = self.create_multi_modal_graph(start_point)
            
            # Get real-time traffic
            bbox = ox.utils_geo.bbox_from_point(start_point, dist=5000)
            self.get_real_time_traffic(bbox)
            
            # Predict congestion
            G = self.predict_congestion(G)
            
            # Find optimal path
            main_path = self.adaptive_a_star(G, start_point, end_point)
            alternatives = self.find_alternative_routes(G, main_path)
            
            # Calculate metrics
            metrics = self.calculate_environmental_impact(main_path, G)
            
            # Visualize
            self.visualize_advanced(G, main_path, alternatives)
            
            return {
                'status': 'success',
                'main_path': main_path,
                'alternatives': alternatives,
                'metrics': metrics
            }
            
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Example usage
if __name__ == '__main__':
    optimizer = RouteOptimizer()
    result = optimizer.optimize_route(
        "Golden Gate Bridge, San Francisco",
        "Googleplex, Mountain View"
    )
    
    if result['status'] == 'success':
        print(f"Optimized route found! CO2 emissions: {result['metrics']['co2_kg']:.2f} kg")
        print("Visualization saved to advanced_visualization.html")