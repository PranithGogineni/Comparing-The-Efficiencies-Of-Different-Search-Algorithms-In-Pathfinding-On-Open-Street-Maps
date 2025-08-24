import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import time

# Step 1: Load a Road Network Focused on Connaught Place
place_name = "Connaught Place, New Delhi, India"
print(f"Loading road network for {place_name}...")
G = ox.graph_from_address(place_name, dist=5000, network_type="drive", simplify=True)
print(f"Graph Loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

# Step 2: Select Start and Goal Nodes (Ensuring Connectivity)

start_lat, start_lon = 28.6483, 77.2097  # Palika Bazaar
goal_lat, goal_lon = 28.6182, 77.2371   # India Gate

print("Finding nearest nodes...")
start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
goal_node = ox.distance.nearest_nodes(G, X=goal_lon, Y=goal_lat)
print(f"Start Node: {start_node}, Goal Node: {goal_node}")

if not nx.has_path(G, start_node, goal_node):
    raise ValueError("No valid path exists between start and goal!")

# Step 3: Implement A* Search with Weakened Heuristic
def weakened_heuristic(u, v):
    """Weakened heuristic to force A* to behave differently than Dijkstra"""
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    
    # Artificially weaken the heuristic by scaling it down significantly
    return 0.1 * ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  

def astar_search(G, start, goal):
    print("Running A* algorithm...")
    start_time = time.time()
    path = nx.astar_path(G, start, goal, weight="length", heuristic=weakened_heuristic)
    end_time = time.time()
    explored_nodes = len(set(path))  # Nodes actually used in the path
    return path, end_time - start_time, explored_nodes

astar_path, astar_time, astar_explored = astar_search(G, start_node, goal_node)
print(f"A* Path Found! Time Taken: {astar_time:.4f} sec, Nodes Explored: {astar_explored}")

# Step 4: Implement Dijkstra Search
def dijkstra_search(G, start, goal):
    print("Running Dijkstra's algorithm...")
    start_time = time.time()
    path = nx.dijkstra_path(G, start, goal, weight="length")
    end_time = time.time()
    explored_nodes = len(set(path))  # Nodes actually used in the path
    return path, end_time - start_time, explored_nodes

dijkstra_path, dijkstra_time, dijkstra_explored = dijkstra_search(G, start_node, goal_node)
print(f"Dijkstra Path Found! Time Taken: {dijkstra_time:.4f} sec, Nodes Explored: {dijkstra_explored}")

# Step 5: Visualize Paths (A* vs. Dijkstra)
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(G, ax=ax, node_size=5, edge_linewidth=0.3, bgcolor="white", show=False, close=False)

# Plot A* and Dijkstra Paths Together
ox.plot_graph_routes(G, [dijkstra_path, astar_path], route_colors=["purple", "blue"], 
                     route_linewidth=3, node_size=50, ax=ax, show=False, close=False)

# Mark Start and Goal Nodes
start_x, start_y = G.nodes[start_node]['x'], G.nodes[start_node]['y']
goal_x, goal_y = G.nodes[goal_node]['x'], G.nodes[goal_node]['y']
ax.scatter(start_x, start_y, c="green", s=100, label="Start")  # Green for start
ax.scatter(goal_x, goal_y, c="red", s=100, label="Goal")  # Red for goal

plt.legend()
plt.show()
