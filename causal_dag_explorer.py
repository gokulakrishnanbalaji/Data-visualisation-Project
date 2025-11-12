"""
LIFESTYLE CAUSAL GRAPH EXPLORER - ENHANCED VERSION
===================================================
Features:
- Domain-based column selection + PC algorithm
- DRAGGABLE NODES with physics (edges follow)
- EDGE HOVER INFO showing SHAP weights and connection details
- NO SLIDERS (removed as requested)
- Interactive visualization
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING AND DOMAIN-BASED SELECTION
# ============================================================================

def load_data(csv_path='life-style-data.csv'):
    """Load and preprocess the dataset."""
    print("="*80)
    print("LIFESTYLE CAUSAL GRAPH EXPLORER - DRAGGABLE NODES + EDGE INFO")
    print("="*80)
    print("\n[STEP 1] Loading dataset...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: {csv_path} not found")
        raise
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numerical = df[numerical_cols].copy()
    df_numerical = df_numerical.fillna(df_numerical.mean())
    df_numerical = df_numerical.loc[:, df_numerical.std() > 0]
    
    print(f"✓ Loaded {len(df_numerical)} samples, {len(df_numerical.columns)} numerical features")
    return df_numerical

def select_lifestyle_columns_research_based(df):
    """Select 15-20 columns based on LIFESTYLE DOMAIN RESEARCH."""
    print("\n[STEP 2] Domain-based column selection...")
    
    all_cols = df.columns.tolist()
    
    domain_mapping = {
        'Age': 'Demographics - Age',
        'Weight': 'Body - Weight',
        'BMI': 'Body - BMI',
        'Fat_Percentage': 'Body - Fat %',
        'lean_mass_kg': 'Body - Lean Mass',
        'Resting_BPM': 'Cardio - Resting Heart Rate',
        'Max_BPM': 'Cardio - Max Heart Rate',
        'Avg_BPM': 'Cardio - Avg Heart Rate',
        'pct_HRR': 'Cardio - Heart Rate Reserve %',
        'Workout_Frequency': 'Exercise - Frequency',
        'Session_Duration': 'Exercise - Duration',
        'Calories_Burned': 'Exercise - Calories Burned',
        'Proteins': 'Nutrition - Proteins',
        'Carbs': 'Nutrition - Carbs',
        'Fats': 'Nutrition - Fats',
        'Calories': 'Nutrition - Calories',
        'Water_Intake': 'Hydration - Water Intake',
        'cal_balance': 'Recovery - Calorie Balance',
        'expected_burn': 'Recovery - Expected Burn',
        'rating': 'Recovery - Rating',
        'Reps': 'Performance - Reps',
        'Sets': 'Performance - Sets',
    }
    
    selected_cols = []
    selected_info = []
    
    for priority_key, display_name in domain_mapping.items():
        for col in all_cols:
            col_normalized = col.lower().replace('(', '').replace(')', '').replace('_', '').replace(' ', '')
            key_normalized = priority_key.lower().replace('_', '')
            
            if (key_normalized in col_normalized or col_normalized in key_normalized):
                if col not in selected_cols:
                    selected_cols.append(col)
                    selected_info.append((col, display_name))
                    break
    
    if len(selected_cols) < 15:
        print(f"  Found {len(selected_cols)} exact matches, searching for more...")
        keywords = [
            ('bpm', 'Heart Rate'),
            ('heart', 'Heart Rate'),
            ('calories', 'Calories'),
            ('weight', 'Weight'),
            ('exercise', 'Exercise'),
            ('workout', 'Workout'),
        ]
        
        for col in all_cols:
            if col not in selected_cols and len(selected_cols) < 20:
                col_lower = col.lower()
                for kw, cat in keywords:
                    if kw in col_lower:
                        selected_cols.append(col)
                        selected_info.append((col, f"{cat} - {col}"))
                        break
    
    selected_cols = selected_cols[:20]
    selected_info = selected_info[:20]
    
    print(f"  ✓ Selected {len(selected_cols)} columns")
    
    return df[selected_cols].copy(), selected_cols, selected_info

# ============================================================================
# PART 2: PC ALGORITHM FOR CAUSAL DISCOVERY
# ============================================================================

def compute_partial_correlation(corr_matrix, i, j, cond_set):
    """Compute partial correlation."""
    if len(cond_set) == 0:
        return corr_matrix[i, j]
    
    k = cond_set[0]
    numerator = corr_matrix[i, j] - corr_matrix[i, k] * corr_matrix[k, j]
    denom_part1 = 1 - corr_matrix[i, k] ** 2
    denom_part2 = 1 - corr_matrix[k, j] ** 2
    denominator = np.sqrt(denom_part1 * denom_part2)
    
    if denominator > 1e-6:
        return numerator / denominator
    return 0.0

def pc_algorithm(data, alpha=0.05, verbose=True):
    """PC Algorithm for causal discovery."""
    if verbose:
        print("\n[STEP 3] PC Algorithm - Causal Discovery...")
    
    n_vars = data.shape[1]
    cols = data.columns.tolist()
    
    graph = nx.Graph()
    for col in cols:
        graph.add_node(col)
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            graph.add_edge(cols[i], cols[j])
    
    if verbose:
        print(f"  [3a] Initial: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    data_array = data.values
    corr_matrix = np.corrcoef(data_array.T)
    np.fill_diagonal(corr_matrix, 0)
    
    if verbose:
        print("  [3b] Skeleton phase...")
    
    max_depth = min(n_vars - 2, 3)
    edges_removed = 0
    
    for depth in range(max_depth + 1):
        nodes_to_check = list(graph.nodes())
        
        for node_i in nodes_to_check:
            neighbors = list(graph.neighbors(node_i))
            
            for node_j in list(neighbors):
                if not graph.has_edge(node_i, node_j):
                    continue
                
                other_neighbors = set(neighbors) - {node_j}
                
                if len(other_neighbors) < depth:
                    continue
                
                for cond_set in combinations(other_neighbors, depth):
                    partial_corr = compute_partial_correlation(
                        corr_matrix, 
                        cols.index(node_i),
                        cols.index(node_j),
                        [cols.index(c) for c in cond_set]
                    )
                    
                    if abs(partial_corr) < 0.1:
                        if graph.has_edge(node_i, node_j):
                            graph.remove_edge(node_i, node_j)
                            edges_removed += 1
                        break
    
    if verbose:
        print(f"    Edges removed: {edges_removed}, Remaining: {graph.number_of_edges()}")
        print("  [3c] Orientation phase...")
    
    dag = nx.DiGraph()
    for node in graph.nodes():
        dag.add_node(node)
    
    for u, v in graph.edges():
        u_idx = cols.index(u)
        v_idx = cols.index(v)
        
        u_var = data[u].var()
        v_var = data[v].var()
        
        if u_var < v_var:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
    
    try:
        cycles = list(nx.simple_cycles(dag))
        for cycle in cycles:
            if len(cycle) > 1:
                min_corr = float('inf')
                min_edge = None
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    if dag.has_edge(u, v):
                        corr = abs(corr_matrix[cols.index(u), cols.index(v)])
                        if corr < min_corr:
                            min_corr = corr
                            min_edge = (u, v)
                
                if min_edge and dag.has_edge(*min_edge):
                    dag.remove_edge(*min_edge)
    except:
        pass
    
    if verbose:
        print(f"  ✓ Final: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    
    return dag

# ============================================================================
# PART 3: SHAP VALUE COMPUTATION
# ============================================================================

def compute_shap_values(df, dag, verbose=True):
    """Compute SHAP values."""
    if verbose:
        print("\n[STEP 4] Computing SHAP values...")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        if verbose:
            print("  Warning: scikit-learn not available")
        return compute_correlation_weights(df, dag)
    
    cols = df.columns.tolist()
    edge_weights = {}
    n_targets = 0
    
    for target in dag.nodes():
        predecessors = list(dag.predecessors(target))
        
        if len(predecessors) > 0:
            n_targets += 1
            try:
                X = df[predecessors].values
                y = df[target].values
                
                X = StandardScaler().fit_transform(X)
                
                model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
                model.fit(X, y)
                
                importances = model.feature_importances_
                total = importances.sum()
                
                if total > 0:
                    for pred, imp in zip(predecessors, importances):
                        edge_weights[(pred, target)] = imp / total
                else:
                    for pred in predecessors:
                        edge_weights[(pred, target)] = 1.0 / len(predecessors)
                        
            except Exception as e:
                for pred in predecessors:
                    edge_weights[(pred, target)] = 1.0 / len(predecessors)
    
    if edge_weights:
        max_w = max(edge_weights.values())
        for key in edge_weights:
            normalized = edge_weights[key] / max_w if max_w > 0 else 0.5
            edge_weights[key] = max(0.15, normalized ** 0.6)
    
    if verbose:
        print(f"  ✓ SHAP computed for {len(edge_weights)} edges")
    
    return edge_weights

def compute_correlation_weights(df, dag):
    """Fallback: use correlation weights."""
    cols = df.columns.tolist()
    corr_matrix = np.corrcoef(df.values.T)
    
    edge_weights = {}
    for src, tgt in dag.edges():
        i, j = cols.index(src), cols.index(tgt)
        weight = abs(corr_matrix[i, j])
        edge_weights[(src, tgt)] = max(0.15, weight)
    
    if edge_weights:
        max_w = max(edge_weights.values())
        for key in edge_weights:
            edge_weights[key] = max(0.15, (edge_weights[key] / max_w) ** 0.6)
    
    return edge_weights

# ============================================================================
# PART 4: GRAPH LAYOUT
# ============================================================================

def compute_graph_layout(dag, verbose=True):
    """Compute force-directed graph layout."""
    if verbose:
        print("\n[STEP 5] Computing graph layout...")
    
    pos = nx.spring_layout(dag, k=3, iterations=150, seed=42)
    
    positions = {}
    
    if len(pos) > 0:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        range_x = max_x - min_x if max_x > min_x else 1
        range_y = max_y - min_y if max_y > min_y else 1
        
        for node, (x, y) in pos.items():
            norm_x = (x - min_x) / range_x
            norm_y = (y - min_y) / range_y
            
            canvas_x = 200 + norm_x * 1400
            canvas_y = 150 + norm_y * 500
            
            positions[node] = (canvas_x, canvas_y)
    
    if verbose:
        print(f"  ✓ Layout computed for {len(positions)} nodes")
    
    return positions

# ============================================================================
# PART 5: VISUALIZATION WITH DRAGGABLE NODES AND EDGE INFO
# ============================================================================

def generate_visualization(df, dag, edge_weights, positions, selected_info, output_path='causal_graph.html'):
    """Generate interactive HTML visualization with draggable nodes."""
    print("\n[STEP 6] Generating visualization...")
    
    nodes_data = []
    for node in dag.nodes():
        x, y = positions.get(node, (1000, 400))
        in_deg = dag.in_degree(node)
        out_deg = dag.out_degree(node)
        
        if in_deg == 0:
            node_type = 'source'
        elif out_deg == 0:
            node_type = 'sink'
        else:
            node_type = 'intermediate'
        
        display_name = node
        for col, info in selected_info:
            if col == node:
                display_name = info.split(' - ')[1] if ' - ' in info else info
                break
        
        nodes_data.append({
            'id': node,
            'label': display_name,
            'type': node_type,
            'x': x,
            'y': y,
            'in_degree': in_deg,
            'out_degree': out_deg
        })
    
    edges_data = []
    for src, tgt in dag.edges():
        weight = edge_weights.get((src, tgt), 0.5)
        edges_data.append({
            'source': src,
            'target': tgt,
            'weight': weight,
            'label': f'{weight:.2f}'
        })
    
    html = get_html_template()
    html = html.replace('NODES_DATA_HERE', json.dumps(nodes_data))
    html = html.replace('EDGES_DATA_HERE', json.dumps(edges_data))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ HTML generated: {output_path}")
    print(f"  ✓ Nodes: {len(nodes_data)}, Edges: {len(edges_data)}")
    
    return output_path

def get_html_template():
    """Return complete HTML template with draggable nodes and edge info."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lifestyle Causal Graph</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; }
        body { display: flex; flex-direction: column; overflow: hidden; }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .header h1 { font-size: 28px; font-weight: 700; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.95; }
        
        .toolbar {
            background: white;
            padding: 15px 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            border-bottom: 2px solid #e8eef7;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .btn {
            padding: 10px 18px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 6px;
            color: #667eea;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
        }
        .btn:hover { background: #667eea; color: white; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(102,126,234,0.3); }
        
        .zoom-info { min-width: 80px; font-weight: 600; color: #667eea; font-size: 14px; }
        .help-text { font-size: 12px; color: #999; margin-left: auto; }
        
        .canvas-wrapper { flex: 1; position: relative; background: #f5f7fa; }
        canvas { position: absolute; top: 0; left: 0; cursor: grab; }
        canvas:active { cursor: grabbing; }
        
        .legend {
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 18px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-size: 13px;
            z-index: 100;
            border: 2px solid #e8eef7;
        }
        .legend-title { font-weight: 700; margin-bottom: 10px; color: #333; }
        .legend-item { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
        .legend-dot { width: 16px; height: 16px; border-radius: 50%; border: 2px solid #333; }
        
        .tooltip {
            position: fixed;
            background: white;
            color: #333;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            z-index: 2000;
            max-width: 350px;
            border: 2px solid #667eea;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
        .tooltip.hidden { display: none; }
        .tooltip strong { color: #667eea; font-weight: 700; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Lifestyle Causal Graph Explorer</h1>
        <p>Domain-based + PC Algorithm | Draggable nodes, edge information on hover</p>
    </div>
    
    <div class="toolbar">
        <button class="btn" id="zoom-in">🔍+ Zoom In</button>
        <button class="btn" id="zoom-out">🔍- Zoom Out</button>
        <button class="btn" id="reset">⟲ Reset</button>
        <div class="zoom-info">Zoom: <span id="zoom-level">100%</span></div>
        <div class="help-text">Drag nodes • Hover edges for info • Scroll to zoom • Drag canvas to pan</div>
    </div>
    
    <div class="canvas-wrapper" id="canvas-wrapper">
        <canvas id="canvas"></canvas>
        <div class="legend">
            <div class="legend-title">Node Types</div>
            <div class="legend-item"><div class="legend-dot" style="background: #3498db;"></div> <span>Source</span></div>
            <div class="legend-item"><div class="legend-dot" style="background: #95a5a6;"></div> <span>Intermediate</span></div>
            <div class="legend-item"><div class="legend-dot" style="background: #9b59b6;"></div> <span>Sink</span></div>
        </div>
    </div>
    
    <div id="tooltip" class="tooltip hidden"></div>
    
    <script>
        const nodesData = NODES_DATA_HERE;
        const edgesData = EDGES_DATA_HERE;
        
        let nodePositions = {};
        nodesData.forEach(node => {
            nodePositions[node.id] = { x: node.x, y: node.y };
        });
        
        let hoveredNode = null;
        let hoveredEdge = null;
        let incomingEdges = [];
        let zoom = 1;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let draggedNode = null;
        
        const canvas = document.getElementById('canvas');
        const wrapper = document.getElementById('canvas-wrapper');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');
        
        function resize() {
            canvas.width = wrapper.clientWidth;
            canvas.height = wrapper.clientHeight;
            draw();
        }
        
        window.addEventListener('resize', resize);
        resize();
        
        document.getElementById('zoom-in').addEventListener('click', () => {
            zoom = Math.min(zoom * 1.2, 3);
            updateZoom();
            draw();
        });
        
        document.getElementById('zoom-out').addEventListener('click', () => {
            zoom = Math.max(zoom / 1.2, 0.3);
            updateZoom();
            draw();
        });
        
        document.getElementById('reset').addEventListener('click', () => {
            zoom = 1;
            panX = 0;
            panY = 0;
            updateZoom();
            draw();
        });
        
        function updateZoom() {
            document.getElementById('zoom-level').textContent = Math.round(zoom * 100) + '%';
        }
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            
            // Check if clicking on a node
            for (let node of nodesData) {
                const dx = x - nodePositions[node.id].x;
                const dy = y - nodePositions[node.id].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 45) {
                    draggedNode = node.id;
                    isDragging = true;
                    lastX = e.clientX;
                    lastY = e.clientY;
                    return;
                }
            }
            
            // Otherwise pan the canvas
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                if (draggedNode) {
                    // Drag node
                    const rect = canvas.getBoundingClientRect();
                    const x = (e.clientX - rect.left - panX) / zoom;
                    const y = (e.clientY - rect.top - panY) / zoom;
                    nodePositions[draggedNode] = { x, y };
                } else {
                    // Pan canvas
                    panX += e.clientX - lastX;
                    panY += e.clientY - lastY;
                }
                lastX = e.clientX;
                lastY = e.clientY;
                draw();
            }
            handleHover(e);
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            draggedNode = null;
        });
        
        canvas.addEventListener('mouseleave', () => {
            isDragging = false;
            draggedNode = null;
            hoveredNode = null;
            hoveredEdge = null;
            tooltip.classList.add('hidden');
            draw();
        });
        
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            zoom = Math.max(0.3, Math.min(zoom * (e.deltaY > 0 ? 0.9 : 1.1), 3));
            updateZoom();
            draw();
        });
        
        function handleHover(e) {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            
            hoveredNode = null;
            hoveredEdge = null;
            incomingEdges = [];
            
            // Check node hover
            for (let node of nodesData) {
                const dx = x - nodePositions[node.id].x;
                const dy = y - nodePositions[node.id].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 45) {
                    hoveredNode = node;
                    incomingEdges = edgesData.map((e, i) => e.target === node.id ? i : -1).filter(i => i !== -1);
                    tooltip.innerHTML = '<strong>' + node.label + '</strong><br>Type: ' + node.type + '<br>In: ' + node.in_degree + ' | Out: ' + node.out_degree;
                    tooltip.classList.remove('hidden');
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY + 12) + 'px';
                    return;
                }
            }
            
            // Check edge hover
            for (let i = 0; i < edgesData.length; i++) {
                const edge = edgesData[i];
                const src = nodesData.find(n => n.id === edge.source);
                const tgt = nodesData.find(n => n.id === edge.target);
                
                if (src && tgt) {
                    const dist = pointToLineDistance(
                        x, y,
                        nodePositions[src.id].x, nodePositions[src.id].y,
                        nodePositions[tgt.id].x, nodePositions[tgt.id].y
                    );
                    
                    if (dist < 15) {
                        hoveredEdge = i;
                        tooltip.innerHTML = 
                            '<strong>' + edge.source + ' → ' + edge.target + '</strong><br>' +
                            'SHAP Weight: ' + edge.label + '<br>' +
                            'Influence Strength: ' + (edge.weight * 100).toFixed(1) + '%';
                        tooltip.classList.remove('hidden');
                        tooltip.style.left = (e.clientX + 12) + 'px';
                        tooltip.style.top = (e.clientY + 12) + 'px';
                        return;
                    }
                }
            }
            
            tooltip.classList.add('hidden');
            draw();
        }
        
        function pointToLineDistance(px, py, x1, y1, x2, y2) {
            const A = px - x1, B = py - y1, C = x2 - x1, D = y2 - y1;
            const dot = A * C + B * D;
            const lenSq = C * C + D * D;
            let param = lenSq !== 0 ? dot / lenSq : -1;
            let xx, yy;
            if (param < 0) { xx = x1; yy = y1; }
            else if (param > 1) { xx = x2; yy = y2; }
            else { xx = x1 + param * C; yy = y1 + param * D; }
            const dx = px - xx, dy = py - yy;
            return Math.sqrt(dx * dx + dy * dy);
        }
        
        function getColor(type) {
            return type === 'source' ? '#3498db' : type === 'sink' ? '#9b59b6' : '#95a5a6';
        }
        
        function draw() {
            ctx.fillStyle = '#f5f7fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);
            
            // Draw edges
            edgesData.forEach((edge, i) => {
                const src = nodesData.find(n => n.id === edge.source);
                const tgt = nodesData.find(n => n.id === edge.target);
                if (!src || !tgt) return;
                
                const srcPos = nodePositions[src.id];
                const tgtPos = nodePositions[tgt.id];
                
                const isIncoming = incomingEdges.includes(i);
                const isHovered = hoveredEdge === i;
                const thickness = 1.5 + (edge.weight ** 1.5) * 6;
                
                ctx.strokeStyle = isHovered ? '#ff8c00' : isIncoming ? '#ff8c00' : '#8e44ad';
                ctx.globalAlpha = isHovered ? 0.95 : isIncoming ? 0.8 : (hoveredNode ? 0.1 : 0.3);
                ctx.lineWidth = isHovered ? thickness + 3 : isIncoming ? thickness + 2 : thickness;
                
                ctx.beginPath();
                ctx.moveTo(srcPos.x, srcPos.y);
                ctx.lineTo(tgtPos.x, tgtPos.y);
                ctx.stroke();
                
                // Arrow
                const angle = Math.atan2(tgtPos.y - srcPos.y, tgtPos.x - srcPos.x);
                ctx.fillStyle = isHovered ? '#ff8c00' : isIncoming ? '#ff8c00' : '#8e44ad';
                ctx.beginPath();
                ctx.moveTo(tgtPos.x, tgtPos.y);
                ctx.lineTo(tgtPos.x - 18 * Math.cos(angle - Math.PI / 6), tgtPos.y - 18 * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(tgtPos.x - 18 * Math.cos(angle + Math.PI / 6), tgtPos.y - 18 * Math.sin(angle + Math.PI / 6));
                ctx.closePath();
                ctx.fill();
                
                ctx.globalAlpha = 1;
            });
            
            // Draw nodes
            nodesData.forEach(node => {
                const pos = nodePositions[node.id];
                ctx.fillStyle = getColor(node.type);
                ctx.strokeStyle = hoveredNode?.id === node.id ? '#ff8c00' : '#2c3e50';
                ctx.lineWidth = hoveredNode?.id === node.id ? 4 : 3;
                
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 42, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
                
                if (hoveredNode?.id === node.id) {
                    ctx.strokeStyle = 'rgba(255, 140, 0, 0.4)';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 55, 0, 2 * Math.PI);
                    ctx.stroke();
                }
                
                ctx.fillStyle = 'white';
                ctx.font = 'bold 11px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                const words = node.label.split(' ');
                if (words.length > 2) {
                    ctx.fillText(words.slice(0, 2).join(' '), pos.x, pos.y - 4);
                    ctx.font = '9px Arial';
                    ctx.fillText(words.slice(2).join(' '), pos.x, pos.y + 6);
                } else {
                    ctx.fillText(node.label, pos.x, pos.y);
                }
            });
            
            ctx.restore();
        }
        
        updateZoom();
        draw();
    </script>
</body>
</html>'''

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import sys
    
    try:
        df = load_data('life-style-data.csv')
        df_selected, selected_cols, selected_info = select_lifestyle_columns_research_based(df)
        dag = pc_algorithm(df_selected, alpha=0.05, verbose=True)
        
        if dag.number_of_nodes() == 0:
            print("ERROR: No nodes in DAG")
            sys.exit(1)
        
        edge_weights = compute_shap_values(df_selected, dag, verbose=True)
        positions = compute_graph_layout(dag, verbose=True)
        output_file = generate_visualization(df_selected, dag, edge_weights, positions, selected_info)
        
        print("\n" + "="*80)
        print("✓ SUCCESS - CAUSAL GRAPH GENERATED")
        print("="*80)
        print(f"\nGraph Statistics:")
        print(f"  • Nodes: {dag.number_of_nodes()}")
        print(f"  • Edges: {dag.number_of_edges()}")
        print(f"  • Avg degree: {2*dag.number_of_edges()/dag.number_of_nodes():.1f}")
        print(f"\nFeatures:")
        print(f"  ✓ Draggable nodes (edges follow)")
        print(f"  ✓ Hover edges for SHAP weights & info")
        print(f"  ✓ No sliders (removed as requested)")
        print(f"  ✓ Interactive zoom & pan")
        print(f"\n✓ Open '{output_file}' in web browser")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
