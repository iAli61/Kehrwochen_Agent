import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np

# For interactive web applications (uncomment if you're using Dash)
# from dash import Dash, dcc, html, Input, Output, clientside_callback, ClientsideFunction

# Load and Preprocess Data

# Load the LeanIX data from an Excel file
# Adjust the path to match your file location
leanix_file_path = 'files/LeanIX_Datenabzug.xlsx'
leanix_data = pd.read_excel(leanix_file_path, header=1)  # Skip the first row (descriptions)

# Clean the data: Remove rows with missing application names or dependencies
leanix_data = leanix_data.dropna(subset=['Name', 'Provided Information Flow/Interface'])

# Extract dependency relationships
def extract_dependencies(df):
    """
    Extracts dependency relationships between applications from the provided DataFrame.
    
    Args:
        df: Pandas DataFrame containing application data with dependency information.
    
    Returns:
        List of tuples representing directed edges (source, target) in the dependency network.
    """
    edges = []
    for _, row in df.iterrows():
        source_app = row['Name']
        if pd.notna(row['Provided Information Flow/Interface']):
            interfaces = row['Provided Information Flow/Interface'].split(';')
            for interface in interfaces:
                if '>>' in interface:
                    parts = interface.split('>>')
                    if len(parts) >= 2:
                        target_app = parts[1].strip().split(' ')[0]
                        edges.append((source_app, target_app))
    return edges

# Create a directed graph using NetworkX
dependency_edges = extract_dependencies(leanix_data)
dependency_graph = nx.DiGraph()
dependency_graph.add_edges_from(dependency_edges)

# Display basic information about the graph
print(f"Number of nodes: {dependency_graph.number_of_nodes()}")
print(f"Number of edges: {dependency_graph.number_of_edges()}")

import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np

def create_enhanced_network_graph(graph):
    """
    Creates an interactive network graph using Plotly with improved node selection functionality.

    Args:
        graph: A NetworkX graph object representing the dependency network.

    Returns:
        A Plotly Figure object for the interactive network graph.
    """
    # Extract node positions using NetworkX's spring layout
    pos = nx.spring_layout(graph, seed=42)

    # Create node and edge traces
    # Create separate edge traces for regular, incoming, and outgoing connections
    edge_x = []
    edge_y = []
    edge_source = []
    edge_target = []
    
    for edge in graph.edges():
        source, target = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Store source and target for each edge segment
        edge_source.extend([source, source, None])
        edge_target.extend([target, target, None])
    
    # Create the edge trace with additional information for highlighting
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color='#888'),
        hoverinfo='none',
        mode='lines',
        opacity=0.6,
        customdata=list(zip(edge_source, edge_target)),  # Store source-target pairs
        name='connections'
    )

    # Create node trace with connection information
    node_x = []
    node_y = []
    node_text = []
    node_ids = []
    node_sizes = []
    node_colors = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get connected nodes
        successors = list(graph.successors(node))
        predecessors = list(graph.predecessors(node))
        
        # Store connection information for hover text
        out_connections = f"Outgoing: {len(successors)}"
        in_connections = f"Incoming: {len(predecessors)}"
        connection_info = f"<br>{out_connections}<br>{in_connections}"
        node_text.append(f"<b>{node}</b>{connection_info}")
        
        # Store node ID
        node_ids.append(node)
        
        # Size nodes based on total connections
        total_connections = len(successors) + len(predecessors)
        node_sizes.append(10 + total_connections * 0.5)  # Base size + scaled by connections
        
        # Color nodes based on ratio of incoming/outgoing
        if total_connections > 0:
            ratio = len(successors) / total_connections if total_connections > 0 else 0.5
            # Color from red (mostly incoming) to blue (mostly outgoing)
            node_colors.append(f'rgb({int(255 * (1-ratio))}, {int(100)}, {int(255 * ratio)})')
        else:
            node_colors.append('rgb(150, 150, 150)')  # Gray for isolated nodes

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white'),
            opacity=0.8,
        ),
        text=node_text,
        customdata=node_ids,  # Store node identifiers
        name='applications',
        # Enable selection of points
        selectedpoints=[],
        selected=dict(
            marker=dict(
                color='rgba(255, 255, 0, 1)',  # Yellow for selected node
                size=16,
            )
        ),
        unselected=dict(
            marker=dict(opacity=0.5)  # Fade unselected nodes
        )
    )
    
    # Create highlighted edge traces (initially empty)
    incoming_edges_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=2.5, color='rgba(255, 0, 0, 0.8)'),  # Red for incoming
        hoverinfo='none',
        mode='lines',
        name='incoming connections',
        visible=True
    )
    
    outgoing_edges_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=2.5, color='rgba(0, 0, 255, 0.8)'),  # Blue for outgoing
        hoverinfo='none',
        mode='lines',
        name='outgoing connections',
        visible=True
    )

    # Create the traces for selected nodes and connected nodes (initially empty)
    # These were missing in the original code
    selected_node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='rgba(255, 255, 0, 1)',  # Yellow for selected node
            size=16,
            line=dict(width=2, color='black'),
            symbol='star'
        ),
        text=[],
        name='selected node',
        visible=True
    )
    
    incoming_node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='rgba(255, 0, 0, 0.8)',  # Red for incoming nodes
            size=14,
            line=dict(width=1.5, color='black'),
            symbol='circle'
        ),
        text=[],
        name='source nodes',
        visible=True
    )
    
    outgoing_node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color='rgba(0, 0, 255, 0.8)',  # Blue for outgoing nodes
            size=14,
            line=dict(width=1.5, color='black'),
            symbol='circle'
        ),
        text=[],
        name='target nodes',
        visible=True
    )

    # Create the figure with all traces
    fig = go.Figure(data=[
        edge_trace,
        incoming_edges_trace, 
        outgoing_edges_trace,
        node_trace,
        selected_node_trace,
        incoming_node_trace,
        outgoing_node_trace
    ],
    layout=go.Layout(
        title=dict(
            text='Interactive Application Dependency Network',
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        clickmode='event',  # We'll handle selection ourselves
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    ))
    
    # Add annotations
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                text="Click on a node to see its connections. Outgoing connections in blue, incoming in red.",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    # Create the custom JavaScript for node selection
    custom_js = """
    function(data) {
        // Get the clicked point data
        if (!data || !data.points || data.points.length === 0) {
            // No point clicked, reset view
            return {
                'data[0].opacity': 0.6,  // Default edge opacity
                'data[1].x': [],  // Clear incoming edges
                'data[1].y': [],
                'data[2].x': [],  // Clear outgoing edges
                'data[2].y': [],
                'data[4].x': [],  // Clear selected node
                'data[4].y': [],
                'data[4].text': [],
                'data[5].x': [],  // Clear incoming nodes
                'data[5].y': [],
                'data[5].text': [],
                'data[6].x': [],  // Clear outgoing nodes
                'data[6].y': [],
                'data[6].text': []
            };
        }
        
        // Assume the click is on the main node trace (index 3)
        const clickedPoint = data.points[0];
        if (clickedPoint.curveNumber !== 3) {
            // Click was not on a node, ignore
            return {};
        }
        
        const selectedNodeId = clickedPoint.customdata;
        const nodeX = clickedPoint.x;
        const nodeY = clickedPoint.y;
        const nodeText = clickedPoint.text;
        
        // Get all node positions and data
        const nodeIds = data.data[3].customdata;
        const nodeXs = data.data[3].x;
        const nodeYs = data.data[3].y;
        const nodeTexts = data.data[3].text;
        
        // Get all edges from the edge trace
        const edgeData = data.data[0].customdata;
        const edgeX = data.data[0].x;
        const edgeY = data.data[0].y;
        
        // Find incoming edges (where target is the selected node)
        const incomingEdgeIndices = [];
        const incomingNodeIds = new Set();
        
        // Find outgoing edges (where source is the selected node)
        const outgoingEdgeIndices = [];
        const outgoingNodeIds = new Set();
        
        // Process edge data to find connections
        for (let i = 0; i < edgeData.length; i++) {
            if (!edgeData[i]) continue; // Skip null entries
            
            const edge = edgeData[i];
            if (edge[1] === selectedNodeId) {
                // This is an incoming edge
                incomingEdgeIndices.push(i);
                incomingNodeIds.add(edge[0]);
            }
            else if (edge[0] === selectedNodeId) {
                // This is an outgoing edge
                outgoingEdgeIndices.push(i);
                outgoingNodeIds.add(edge[1]);
            }
        }
        
        // Extract edge segments for highlighting
        let incomingEdgeX = [];
        let incomingEdgeY = [];
        for (let i = 0; i < incomingEdgeIndices.length; i++) {
            const idx = incomingEdgeIndices[i];
            // Each edge consists of 3 points (start, end, null)
            const segmentStartIdx = Math.floor(idx / 3) * 3;
            incomingEdgeX.push(edgeX[segmentStartIdx], edgeX[segmentStartIdx + 1], null);
            incomingEdgeY.push(edgeY[segmentStartIdx], edgeY[segmentStartIdx + 1], null);
        }
        
        let outgoingEdgeX = [];
        let outgoingEdgeY = [];
        for (let i = 0; i < outgoingEdgeIndices.length; i++) {
            const idx = outgoingEdgeIndices[i];
            const segmentStartIdx = Math.floor(idx / 3) * 3;
            outgoingEdgeX.push(edgeX[segmentStartIdx], edgeX[segmentStartIdx + 1], null);
            outgoingEdgeY.push(edgeY[segmentStartIdx], edgeY[segmentStartIdx + 1], null);
        }
        
        // Find positions of connected nodes
        let incomingNodeX = [];
        let incomingNodeY = [];
        let incomingNodeText = [];
        
        let outgoingNodeX = [];
        let outgoingNodeY = [];
        let outgoingNodeText = [];
        
        for (let i = 0; i < nodeIds.length; i++) {
            const nodeId = nodeIds[i];
            if (incomingNodeIds.has(nodeId)) {
                incomingNodeX.push(nodeXs[i]);
                incomingNodeY.push(nodeYs[i]);
                incomingNodeText.push(nodeTexts[i] + '<br><b>Sends to selected node</b>');
            }
            else if (outgoingNodeIds.has(nodeId)) {
                outgoingNodeX.push(nodeXs[i]);
                outgoingNodeY.push(nodeYs[i]);
                outgoingNodeText.push(nodeTexts[i] + '<br><b>Receives from selected node</b>');
            }
        }
        
        // Return updates for all traces
        return {
            'data[0].opacity': 0.2,  // Fade regular edges
            'data[1].x': incomingEdgeX,  // Update incoming edges
            'data[1].y': incomingEdgeY,
            'data[2].x': outgoingEdgeX,  // Update outgoing edges
            'data[2].y': outgoingEdgeY,
            'data[4].x': [nodeX],  // Update selected node
            'data[4].y': [nodeY],
            'data[4].text': [nodeText + '<br><b>SELECTED</b>'],
            'data[5].x': incomingNodeX,  // Update incoming nodes
            'data[5].y': incomingNodeY,
            'data[5].text': incomingNodeText,
            'data[6].x': outgoingNodeX,  // Update outgoing nodes
            'data[6].y': outgoingNodeY,
            'data[6].text': outgoingNodeText
        };
    }
    """
    
    # Add reset button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{'clickmode': 'event'}],
                        label="Reset Selection",
                        method="relayout"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    return fig, custom_js

# Generate the enhanced network graph
network_fig, custom_js = create_enhanced_network_graph(dependency_graph)

# If using this in a standalone script, save the figure to an HTML file with proper callbacks
# First we need to add client-side callback functionality to make the selection work
# We'll include a small JavaScript function to handle click events

network_fig.write_html(
    "interactive_dependency_network.html", 
    include_plotlyjs=True,
    full_html=True,
    config={
        'responsive': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['resetScale2d']
    },
    post_script=[
        # Add a callback to handle node selection
        """
        <script>
        const graphDiv = document.getElementById('{}');
        
        // Add click handler
        graphDiv.on('plotly_click', function(data) {
            if (!data || !data.points || data.points.length === 0) return;
            
            const updateData = (""" + custom_js + """)(data);
            Plotly.update(graphDiv, updateData);
        });
        
        // Add button handler for reset
        graphDiv.on('plotly_relayout', function() {
            // Clear all highlights
            const resetData = {
                'data[0].opacity': 0.6,  // Default edge opacity
                'data[1].x': [],  // Clear incoming edges
                'data[1].y': [],
                'data[2].x': [],  // Clear outgoing edges
                'data[2].y': [],
                'data[4].x': [],  // Clear selected node
                'data[4].y': [],
                'data[4].text': [],
                'data[5].x': [],  // Clear incoming nodes
                'data[5].y': [],
                'data[5].text': [],
                'data[6].x': [],  // Clear outgoing nodes
                'data[6].y': [],
                'data[6].text': []
            };
            Plotly.update(graphDiv, resetData);
        });
        </script>
        """
    ]
)

print("Interactive network visualization saved to 'interactive_dependency_network.html'")

# If you're using Dash, here's how to set up the application:
"""
from dash import Dash, html, dcc, Input, Output, clientside_callback

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Application Dependency Network"),
    html.P("Click on a node to see its incoming and outgoing connections"),
    dcc.Graph(
        id='dependency-network',
        figure=network_fig,
        style={'width': '100%', 'height': '80vh'}
    )
])

# Add a ClientsideFunction to handle node selection
app.clientside_callback(

    function(clickData) {
        const graphDiv = document.getElementById('dependency-network');
        if (!clickData) {
            // Reset view if no point is clicked
            const resetData = {
                'data[0].opacity': 0.6,
                'data[1].x': [],
                'data[1].y': [],
                'data[2].x': [],
                'data[2].y': [],
                'data[4].x': [],
                'data[4].y': [],
                'data[4].text': [],
                'data[5].x': [],
                'data[5].y': [],
                'data[5].text': [],
                'data[6].x': [],
                'data[6].y': [],
                'data[6].text': []
            };
            return resetData;
        }
        
        // Use the custom selection function
        return %s(clickData);
    }
    Output('dependency-network', 'figure'),
    Input('dependency-network', 'clickData')
)

if __name__ == '__main__':
    app.run_server(debug=True)
"""