import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback
import sys
import warnings

# For styles management
import os

# Display informative messages
print("Starting Application Portfolio Dashboard...")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Data preprocessing
def preprocess_data(df):
    # Create copies of the columns and fill nulls with 'unclassified'
    df['Business Value (classified)'] = df['Business Value'].fillna('unclassified')
    df['Risk / Complexity (classified)'] = df['Risk / Complexity'].fillna('unclassified')
    df['Lifecycle Status'] = df['Lifecycle: Current Lifecycle'].fillna('unclassified')
    
    # Create mappings for numeric positions including unclassified
    df['Value_numeric'] = df['Business Value (classified)'].map({
        'low': 1, 'medium': 2, 'high': 3, 'veryHigh': 4, 'unclassified': 0
    })
    
    df['Risk_numeric'] = df['Risk / Complexity (classified)'].map({
        'low': 1, 'medium': 2, 'high': 3, 'veryHigh': 4, 'unclassified': 0
    })
    
    # Add jitter to prevent perfect overlaps
    df['Value_jittered'] = df['Value_numeric'] + (np.random.rand(len(df)) - 0.5) * 0.2
    df['Risk_jittered'] = df['Risk_numeric'] + (np.random.rand(len(df)) - 0.5) * 0.2
    
    # Add strategic quadrant classification
    def get_strategic_quadrant(row):
        value = row['Business Value (classified)']
        risk = row['Risk / Complexity (classified)']
        
        if value == 'unclassified' or risk == 'unclassified':
            return 'Unclassified'
        
        value_level = {'low': 1, 'medium': 1, 'high': 2, 'veryHigh': 2}.get(value, 0)
        risk_level = {'low': 1, 'medium': 1, 'high': 2, 'veryHigh': 2}.get(risk, 0)
        
        if value_level == 2 and risk_level == 1:
            return 'Strategic (Invest & Grow)'
        elif value_level == 1 and risk_level == 1:
            return 'Utility (Maintain Efficiently)'
        elif value_level == 1 and risk_level == 2:
            return 'Questionable (Consider Eliminating)'
        elif value_level == 2 and risk_level == 2:
            return 'Critical (Manage Risks)'
        else:
            return 'Unclassified'
    
    df['Strategic Quadrant'] = df.apply(get_strategic_quadrant, axis=1)
    
    # Define color mapping for strategic quadrants
    df['Quadrant_color'] = df['Strategic Quadrant'].map({
        'Strategic (Invest & Grow)': '#4CAF50',        # Green
        'Utility (Maintain Efficiently)': '#2196F3',   # Blue
        'Questionable (Consider Eliminating)': '#F44336',  # Red
        'Critical (Manage Risks)': '#FFC107',          # Amber
        'Unclassified': '#9E9E9E'                      # Grey
    })
    
    return df

# Create the Dash application
app = Dash(__name__, suppress_callback_exceptions=True)

# Create assets folder for CSS if it doesn't exist
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')):
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))

# Write custom CSS to a file in the assets folder
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'custom.css'), 'w') as f:
    f.write('''
/* Table styles */
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f2f2f2;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Summary section styles */
.summary-section {
    margin-bottom: 15px;
    padding: 10px;
    background-color: #f5f5f5;
    border-radius: 5px;
}

.summary-header {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ddd;
}
''')

# Define the application layout
app.layout = html.Div([
    html.H1("Interactive Application Portfolio Dashboard"),
    
    # Top stats row
    html.Div([
        html.Div([
            html.H3("Portfolio Summary"),
            html.Div(id='portfolio-summary')
        ], style={'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'padding': '15px', 
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '0 10px'})
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
    
    # Main content area
    html.Div([
        # Left column - plot
        html.Div([
            dcc.Graph(id='portfolio-plot'),
        ], style={'padding': '10px', 'flex': '60%'}),
        
        # Right column - details
        html.Div([
            html.H3("Selection Details"),
            html.Div(id='selected-info'),
            html.Div(id='app-table-container')
        ], style={'padding': '10px', 'flex': '40%', 'maxHeight': '800px', 'overflowY': 'auto'})
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Store for the processed data
    dcc.Store(id='processed-data'),
    
    # Store for the currently selected point
    dcc.Store(id='selected-point', data={'x': None, 'y': None}),
    
    # Store for the styling
    dcc.Store(
        id='css-store',
        data={
            'styles': {
                'main-content': {
                    'display': 'flex',
                    'flexWrap': 'wrap'
                },
                'column': {
                    'padding': '10px'
                },
                'left-column': {
                    'flex': '60%'
                },
                'right-column': {
                    'flex': '40%',
                    'maxHeight': '800px',
                    'overflowY': 'auto'
                },
                'stats-row': {
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-between',
                    'marginBottom': '20px'
                },
                'stats-card': {
                    'backgroundColor': '#f9f9f9',
                    'borderRadius': '5px',
                    'padding': '15px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'flex': '1',
                    'margin': '0 10px'
                }
            }
        }
    )
])

# Define the file path for LeanIX data
leanix_file_path = 'files/LeanIX_Datenabzug.xlsx'

# Function to load and preprocess LeanIX data
def load_leanix_data(file_path):
    try:
        # Import library for Excel files
        import openpyxl
        from openpyxl import load_workbook
        
        print(f"Loading LeanIX data from: {file_path}")
        
        # Load Excel file using pandas
        leanix_data = pd.read_excel(file_path, header=1)
        
        print(f"Successfully loaded data with {len(leanix_data)} rows")
        
        # Basic data cleaning
        # Convert column names to match expected format if needed
        # This may need adjustment based on actual LeanIX export format
        column_mapping = {
            # Map your actual column names to the expected ones
            # Example: 'Original Column Name': 'Expected Column Name'
        }
        
        # Rename columns if mapping is provided
        if column_mapping:
            leanix_data.rename(columns=column_mapping, inplace=True)
        
        # Ensure required columns exist
        required_columns = [
            'Business Value', 
            'Risk / Complexity', 
            'Lifecycle: Current Lifecycle',
            'Subtype',
            'Which type of application architecture is used?',
            'TIME Classification'
        ]
        
        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in leanix_data.columns]
        if missing_cols:
            print(f"Warning: The following required columns are missing: {missing_cols}")
            print("Available columns:", leanix_data.columns.tolist())
            
            # Create missing columns with NaN values
            for col in missing_cols:
                leanix_data[col] = np.nan
        
        return leanix_data
        
    except Exception as e:
        print(f"Error loading LeanIX data: {str(e)}")
        # Return empty DataFrame if there's an error
        return pd.DataFrame()

# Callback to store processed data
@app.callback(
    Output('processed-data', 'data'),
    Input('processed-data', 'id')
)
def process_data(_):
    try:
        # Load data from LeanIX Excel file
        leanix_data = load_leanix_data(leanix_file_path)  # Skip the first row (descriptions)
        
        # If data loading failed, use sample data as fallback
        if leanix_data.empty:
            print("Using sample data as fallback since LeanIX data could not be loaded")
            
            # Create a sample dataframe with 100 applications
            np.random.seed(42)
            
            # Generate sample data
            n_samples = 100
            business_values = np.random.choice(['low', 'medium', 'high', 'veryHigh', np.nan], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
            risks = np.random.choice(['low', 'medium', 'high', 'veryHigh', np.nan], n_samples, p=[0.15, 0.3, 0.35, 0.1, 0.1])
            lifecycles = np.random.choice(['active', 'phaseOut', 'endOfLife', 'plan', np.nan], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05])
            
            # Create app names and IDs
            app_ids = [f"APP-{i:05d}" for i in range(1, n_samples+1)]
            app_names = [f"Application {i}" for i in range(1, n_samples+1)]
            
            # Create architecture types
            arch_types = np.random.choice(['monolithic', 'modular', 'microservice'], n_samples, p=[0.5, 0.3, 0.2])
            
            # Create subtypes
            subtypes = np.random.choice(['mbgCatBusinessApp', 'mbgCatTechnicalApp', 'mbgCatIDPApp', 'mgbCatExternalApp'], 
                                      n_samples, p=[0.6, 0.25, 0.1, 0.05])
            
            # Create TIME classifications
            time_classes = np.random.choice(['Tolerate', 'Invest', 'Migrate', 'Eliminate'], n_samples, p=[0.3, 0.2, 0.3, 0.2])
            
            # Create sample dataframe
            leanix_data = pd.DataFrame({
                'ID': app_ids,
                'Name': app_names,
                'Business Value': business_values,
                'Risk / Complexity': risks,
                'Lifecycle: Current Lifecycle': lifecycles,
                'Subtype': subtypes,
                'Which type of application architecture is used?': arch_types,
                'TIME Classification': time_classes,
                'Description': [f"Description for {name}" for name in app_names]
            })
        
        # Preprocess the data
        processed_df = preprocess_data(leanix_data)
        
        # Convert to dict for storage in dcc.Store
        return processed_df.to_dict('records')
    
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        return []

# Callback to create the portfolio plot
@app.callback(
    Output('portfolio-plot', 'figure'),
    Input('processed-data', 'data')
)
def create_portfolio_plot(data):
    if not data:
        return go.Figure()
    
    # Convert stored data back to dataframe
    df = pd.DataFrame(data)
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add quadrant background areas
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=2.5, y1=2.5, 
                 fillcolor="#2196F3", opacity=0.1, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=2.5, y0=0.5, x1=4.5, y1=2.5, 
                 fillcolor="#4CAF50", opacity=0.1, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.5, y0=2.5, x1=2.5, y1=4.5, 
                 fillcolor="#F44336", opacity=0.1, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=2.5, y0=2.5, x1=4.5, y1=4.5, 
                 fillcolor="#FFC107", opacity=0.1, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=-0.5, y0=-0.5, x1=0.5, y1=4.5, 
                 fillcolor="#9E9E9E", opacity=0.1, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.5, y0=-0.5, x1=4.5, y1=0.5, 
                 fillcolor="#9E9E9E", opacity=0.1, line=dict(width=0), layer="below")
    
    # Add scatter plot for each lifecycle status
    lifecycle_markers = {
        'active': 'circle',
        'phaseOut': 'triangle-up',
        'endOfLife': 'square',
        'plan': 'diamond',
        'unclassified': 'x'
    }
    
    for lifecycle, marker in lifecycle_markers.items():
        lifecycle_df = df[df['Lifecycle Status'] == lifecycle]
        
        if not lifecycle_df.empty:
            fig.add_trace(go.Scatter(
                x=lifecycle_df['Value_jittered'],
                y=lifecycle_df['Risk_jittered'],
                mode='markers',
                marker=dict(
                    symbol=marker,
                    size=12,
                    color=lifecycle_df['Quadrant_color'],
                    line=dict(width=1, color='black')
                ),
                name=lifecycle,
                text=lifecycle_df['Name'],
                customdata=pd.DataFrame({
                    'ID': lifecycle_df['ID'],
                    'Name': lifecycle_df['Name'],
                    'Value': lifecycle_df['Business Value (classified)'],
                    'Risk': lifecycle_df['Risk / Complexity (classified)'],
                    'Lifecycle': lifecycle_df['Lifecycle Status'],
                    'ValueNum': lifecycle_df['Value_numeric'],
                    'RiskNum': lifecycle_df['Risk_numeric'],
                    'Quadrant': lifecycle_df['Strategic Quadrant']
                }).to_numpy(),
                hovertemplate='<b>%{text}</b><br>' +
                              'ID: %{customdata[0]}<br>' +
                              'Value: %{customdata[2]}<br>' +
                              'Risk: %{customdata[3]}<br>' +
                              'Lifecycle: %{customdata[4]}<br>' +
                              'Quadrant: %{customdata[7]}<br>' +
                              '<extra></extra>'
            ))
    
    # Add quadrant labels
    quadrant_labels = [
        {"x": 1.5, "y": 1.5, "text": "UTILITY<br>(Maintain Efficiently)", "color": "#2196F3"},
        {"x": 3.5, "y": 1.5, "text": "STRATEGIC<br>(Invest & Grow)", "color": "#4CAF50"},
        {"x": 1.5, "y": 3.5, "text": "QUESTIONABLE<br>(Consider Eliminating)", "color": "#F44336"},
        {"x": 3.5, "y": 3.5, "text": "CRITICAL<br>(Manage Risks)", "color": "#FFC107"},
        {"x": 0, "y": 2.5, "text": "UNCLASSIFIED<br>(Requires Assessment)", "color": "#9E9E9E"}
    ]
    
    for label in quadrant_labels:
        fig.add_annotation(
            x=label["x"], y=label["y"],
            text=label["text"],
            font=dict(color=label["color"], size=14),
            showarrow=False,
            align="center",
            bgcolor="white",
            bordercolor=label["color"],
            borderwidth=2,
            borderpad=4,
            opacity=0.8
        )
    
    # Add dividing lines
    fig.add_shape(type="line", x0=2.5, y0=-0.5, x1=2.5, y1=4.5, 
                 line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, y0=2.5, x1=4.5, y1=2.5, 
                 line=dict(color="gray", width=1, dash="dash"))
    
    # Configure axes
    fig.update_layout(
        title="Application Portfolio: Business Value vs. Risk with Lifecycle Breakdown",
        xaxis=dict(
            title="Business Value",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['Unclassified', 'Low', 'Medium', 'High', 'Very High'],
            range=[-0.5, 4.5]
        ),
        yaxis=dict(
            title="Risk / Complexity",
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['Unclassified', 'Low', 'Medium', 'High', 'Very High'],
            range=[-0.5, 4.5]
        ),
        legend=dict(
            title="Lifecycle Status",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        clickmode='event+select',
        hovermode='closest',
        height=700
    )
    
    return fig

# Callback to update portfolio summary stats
@app.callback(
    Output('portfolio-summary', 'children'),
    Input('processed-data', 'data')
)
def update_portfolio_summary(data):
    if not data:
        return "No data available"
    
    df = pd.DataFrame(data)
    
    # Calculate counts
    total_count = len(df)
    active_count = len(df[df['Lifecycle Status'] == 'active'])
    phaseout_count = len(df[df['Lifecycle Status'] == 'phaseOut'])
    endoflife_count = len(df[df['Lifecycle Status'] == 'endOfLife'])
    plan_count = len(df[df['Lifecycle Status'] == 'plan'])
    
    # Calculate data completeness
    unclassified_value = len(df[df['Business Value (classified)'] == 'unclassified'])
    unclassified_risk = len(df[df['Risk / Complexity (classified)'] == 'unclassified'])
    unclassified_lifecycle = len(df[df['Lifecycle Status'] == 'unclassified'])
    
    value_pct = (total_count - unclassified_value) / total_count * 100
    risk_pct = (total_count - unclassified_risk) / total_count * 100
    lifecycle_pct = (total_count - unclassified_lifecycle) / total_count * 100
    
    # Count by quadrant
    quadrant_counts = df['Strategic Quadrant'].value_counts()
    
    return html.Div([
        html.Div([
            html.H4("Application Count"),
            html.P(f"Total: {total_count} applications"),
            html.Div([
                html.Div([
                    html.Strong("Active: "),
                    html.Span(f"{active_count} ({active_count/total_count:.1%})")
                ]),
                html.Div([
                    html.Strong("Phase Out: "),
                    html.Span(f"{phaseout_count} ({phaseout_count/total_count:.1%})")
                ]),
                html.Div([
                    html.Strong("End of Life: "),
                    html.Span(f"{endoflife_count} ({endoflife_count/total_count:.1%})")
                ]),
                html.Div([
                    html.Strong("Planned: "),
                    html.Span(f"{plan_count} ({plan_count/total_count:.1%})")
                ])
            ])
        ], className="summary-section"),
        
        html.Div([
            html.H4("Data Completeness"),
            html.Div([
                html.Div([
                    html.Strong("Business Value: "),
                    html.Span(f"{value_pct:.1f}%")
                ]),
                html.Div([
                    html.Strong("Risk/Complexity: "),
                    html.Span(f"{risk_pct:.1f}%")
                ]),
                html.Div([
                    html.Strong("Lifecycle: "),
                    html.Span(f"{lifecycle_pct:.1f}%")
                ])
            ])
        ], className="summary-section"),
        
        html.Div([
            html.H4("Strategic Positioning"),
            html.Div([
                html.Div([
                    html.Strong(f"{quadrant}: "),
                    html.Span(f"{count} ({count/total_count:.1%})")
                ]) for quadrant, count in quadrant_counts.items()
            ])
        ], className="summary-section")
    ])

# Callback to handle plot click events
@app.callback(
    [Output('selected-point', 'data'),
     Output('selected-info', 'children'),
     Output('app-table-container', 'children')],
    [Input('portfolio-plot', 'clickData')],
    [State('processed-data', 'data'),
     State('selected-point', 'data')]
)
def display_click_data(clickData, data, selected_point):
    if not clickData or not data:
        return selected_point, "Select a point to see details", "No selection"
    
    # Extract click coordinates
    point = clickData['points'][0]
    x = point['customdata'][5]  # Value_numeric
    y = point['customdata'][6]  # Risk_numeric
    
    # Convert to integers to match exact coordinates
    x_int, y_int = int(round(x)), int(round(y))
    
    # Update selection only if we clicked a new point
    if selected_point and selected_point['x'] == x_int and selected_point['y'] == y_int:
        new_selected = selected_point
    else:
        new_selected = {'x': x_int, 'y': y_int}
    
    # Convert back to dataframe
    df = pd.DataFrame(data)
    
    # Find all apps in this position (exact match on numeric value and risk)
    position_df = df[(df['Value_numeric'] == x_int) & (df['Risk_numeric'] == y_int)]
    
    if position_df.empty:
        return new_selected, "No applications found at this position", "No data available"
    
    # Create summary for this position
    value_label = position_df['Business Value (classified)'].iloc[0]
    risk_label = position_df['Risk / Complexity (classified)'].iloc[0]
    quadrant = position_df['Strategic Quadrant'].iloc[0]
    
    # Count apps by lifecycle
    lifecycle_counts = position_df['Lifecycle Status'].value_counts()
    
    # Compute architecture type distribution
    arch_type_counts = position_df['Which type of application architecture is used?'].value_counts()
    
    # Time classification distribution
    time_counts = position_df['TIME Classification'].value_counts()
    
    # Create the summary information
    summary_info = html.Div([
        html.Div([
            html.H4(f"Position: {value_label} Value, {risk_label} Risk"),
            html.P(f"Strategic Quadrant: {quadrant}"),
            html.P(f"Total Applications: {len(position_df)}")
        ], className="summary-header"),
        
        html.Div([
            html.H5("Lifecycle Breakdown"),
            html.Ul([
                html.Li(f"{lifecycle}: {count} ({count/len(position_df):.1%})") 
                for lifecycle, count in lifecycle_counts.items()
            ])
        ], className="summary-section"),
        
        html.Div([
            html.H5("Architecture Types"),
            html.Ul([
                html.Li(f"{arch_type}: {count} ({count/len(position_df):.1%})") 
                for arch_type, count in arch_type_counts.items()
            ]) if not arch_type_counts.empty else html.P("No architecture type data available")
        ], className="summary-section"),
        
        html.Div([
            html.H5("TIME Classification"),
            html.Ul([
                html.Li(f"{time_class}: {count} ({count/len(position_df):.1%})") 
                for time_class, count in time_counts.items()
            ]) if not time_counts.empty else html.P("No TIME classification data available")
        ], className="summary-section")
    ])
    
    # Create the detailed applications table
    table_columns = ['ID', 'Name', 'Lifecycle Status', 'Which type of application architecture is used?', 
                    'TIME Classification', 'Subtype']
    
    # Create a DataFrame with just the columns we want to display
    display_df = position_df[table_columns].copy()
    
    # Rename columns for better readability
    display_df.columns = ['ID', 'Name', 'Lifecycle', 'Architecture', 'TIME Class', 'Subtype']
    
    app_table = html.Div([
        html.H4("Applications"),
        dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in display_df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])
    
    return new_selected, summary_info, app_table

# Main entry point
if __name__ == '__main__':
    # Fixed line: changed from app.run_server(debug=True) to app.run(debug=True)
    app.run(debug=True)