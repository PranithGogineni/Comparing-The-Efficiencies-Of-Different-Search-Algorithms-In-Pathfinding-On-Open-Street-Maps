import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import time
import heapq

# Step 1: Load a Road Network Focused on New York City
place_name = "New York, USA"
print(f"Loading road network for {place_name}...")
G = ox.graph_from_address(place_name, dist=8000, network_type="drive", simplify=True)
print(f"Graph Loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

# Step 2: Select Start and Goal Nodes 
start_lat, start_lon = 40.7580, -73.9855  # Times Square
goal_lat, goal_lon = 40.6892, -74.0445   # Statue of Liberty

print("Finding nearest nodes...")
start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
goal_node = ox.distance.nearest_nodes(G, X=goal_lon, Y=goal_lat)
print(f"Start Node: {start_node}, Goal Node: {goal_node}")

if not nx.has_path(G, start_node, goal_node):
    raise ValueError("No valid path exists between start and goal!")

# Step 3: A* Search
def the_heuristic(u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return 0.9 * ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # euclidean distance formula

def astar_search(G, start, goal):
    print("Running A* algorithm...")
    start_time = time.time()
    path = nx.astar_path(G, start, goal, weight="length", heuristic=the_heuristic)
    end_time = time.time()
    explored_nodes = len(set(path))
    return path, end_time - start_time, explored_nodes

astar_path, astar_time, astar_explored = astar_search(G, start_node, goal_node)
print(f"A* Path Found! Time Taken: {astar_time:.4f} sec, Nodes Explored: {astar_explored}")

# Step 4: Dijkstra Search
def dijkstra_search(G, start, goal):
    print("Running Dijkstra's algorithm...")
    start_time = time.time()
    path = nx.dijkstra_path(G, start, goal, weight="length")
    end_time = time.time()
    explored_nodes = len(set(path))
    return path, end_time - start_time, explored_nodes

dijkstra_path, dijkstra_time, dijkstra_explored = dijkstra_search(G, start_node, goal_node)
print(f"Dijkstra Path Found! Time Taken: {dijkstra_time:.4f} sec, Nodes Explored: {dijkstra_explored}")

# Step 5: Implement BFS Search
def bfs_search(G, start, goal):
    print("Running BFS algorithm...")
    start_time = time.time()
    
    visited = set()
    queue = [start]
    came_from = {start: None}

    while queue:
        current = queue.pop(0)
        visited.add(current)
        
        if current == goal:
            break
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited and neighbor not in queue:
                came_from[neighbor] = current
                queue.append(neighbor)
    
    end_time = time.time()
    
    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node, None)
    path.reverse()

    return path, end_time - start_time, len(visited)

# Call the BFS function (this was missing!)
bfs_path, bfs_time, bfs_explored = bfs_search(G, start_node, goal_node)
print(f"BFS Path Found! Time Taken: {bfs_time:.4f} sec, Nodes Explored: {bfs_explored}")

# Step 6: Implement Greedy Best-First Search
def euclidean_heuristic(u, v):
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  

def greedy_best_first_search(G, start, goal):
    print("Running Greedy Best-First Search algorithm...")
    start_time = time.time()
    
    frontier = [(euclidean_heuristic(start, goal), start)]
    came_from = {start: None}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break  

        for neighbor in G.neighbors(current):
            if neighbor not in came_from:  
                came_from[neighbor] = current
                heapq.heappush(frontier, (euclidean_heuristic(neighbor, goal), neighbor))

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node, None)
    path.reverse()

    end_time = time.time()
    
    if len(path) == 1:
        raise ValueError("No Greedy Best-First path found!")

    explored_nodes = len(came_from)
    return path, end_time - start_time, explored_nodes

gbfs_path, gbfs_time, gbfs_explored = greedy_best_first_search(G, start_node, goal_node)
print(f"Greedy Best-First Path Found! Time Taken: {gbfs_time:.4f} sec, Nodes Explored: {gbfs_explored}")

# Step 7: Visualize Paths (A* vs. Dijkstra vs. BFS vs. GBFS)
fig, ax = plt.subplots(figsize=(12, 12))
ox.plot_graph(G, ax=ax, node_size=5, edge_linewidth=0.3, bgcolor="white", show=False, close=False)

# Plot paths with different colors
ox.plot_graph_routes(G, [bfs_path, dijkstra_path, astar_path, gbfs_path], 
                     route_colors=["red", "purple", "blue", "green"],  
                     route_linewidth=3, node_size=50, ax=ax, show=False, close=False)

# Mark Start and Goal Nodes
start_x, start_y = G.nodes[start_node]['x'], G.nodes[start_node]['y']
goal_x, goal_y = G.nodes[goal_node]['x'], G.nodes[goal_node]['y']
ax.scatter(start_x, start_y, c="green", s=100, label="Start")  
ax.scatter(goal_x, goal_y, c="black", s=100, label="Goal")  

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='red', lw=3, label='BFS'),
    Line2D([0], [0], color='purple', lw=3, label="Dijkstra"),
    Line2D([0], [0], color='blue', lw=3, label="A*"),
    Line2D([0], [0], color='green', lw=3, label="GBFS"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Goal')
]

ax.legend(handles=legend_elements)

plt.show()

# Step 8: Print Comparison Table
print("\n--- A* vs Dijkstra vs BFS vs GBFS ---")
print(f"A* Search: Time = {astar_time:.4f} sec, Nodes Explored = {astar_explored}")
print(f"Dijkstra: Time = {dijkstra_time:.4f} sec, Nodes Explored = {dijkstra_explored}")
print(f"BFS: Time = {bfs_time:.4f} sec, Nodes Explored = {bfs_explored}")
print(f"Greedy Best-First: Time = {gbfs_time:.4f} sec, Nodes Explored = {gbfs_explored}")
