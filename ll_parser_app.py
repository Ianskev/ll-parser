import streamlit as st
import tempfile
import os
import base64
import shutil
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
from LL_parser import parse_grammar_and_analyze, eliminate_left_recursion, left_factorization, generate_parse_tree

# Helper functions to parse the analysis output
def parse_symbol_table(result_text):
    """Extract and parse the symbol table from the result text"""
    # Find the table of symbols section
    symbol_match = re.search(r"TABLA DE SÍMBOLOS\n(.*?)(?=\n\n)", result_text, re.DOTALL)
    if not symbol_match:
        return None
    
    symbol_table_text = symbol_match.group(1)
    lines = symbol_table_text.strip().split('\n')
    
    # Extract header
    header_match = re.match(r"Símbolo\s+Tipo\s+NULLABLE\s+FIRST\s+FOLLOW", lines[0])
    if not header_match:
        # Try to find header line
        for i, line in enumerate(lines):
            if "Símbolo" in line and "Tipo" in line and "NULLABLE" in line:
                header_line = i
                break
        else:
            return None
    
    # Skip header and separator lines
    data_lines = [line for line in lines[2:] if line.strip() and not all(c == '-' for c in line.strip())]
    
    # Parse data lines
    data = []
    for line in data_lines:
        # Split by multiple spaces and filter out empty strings
        parts = [p for p in re.split(r'\s{2,}', line.strip()) if p]
        if len(parts) >= 5:
            # Format FIRST and FOLLOW with curly braces if they're not empty
            first = parts[3]
            follow = parts[4]
            
            if first and first != "-":
                first = "{" + first + "}"
            
            if follow and follow != "-":
                follow = "{" + follow + "}"
                
            data.append({
                "Símbolo": parts[0],
                "Tipo": parts[1],
                "FIRST": first,
                "FOLLOW": follow
            })
    
    return pd.DataFrame(data)

def parse_chain_analysis(result_text):
    """Extract and parse the chain analysis from the result text"""
    # Find the analysis chain section
    analysis_match = re.search(r"ANALISIS DE LA CADENA\n(.*?)(?=\n\nResultado)", result_text, re.DOTALL)
    if not analysis_match:
        return None
    
    analysis_text = analysis_match.group(1)
    lines = analysis_text.strip().split('\n')
    
    # Extract header and find header positions
    header_line = lines[0]
    sep_line = lines[1]
    
    # Parse header columns based on the separator line
    col_positions = [0]
    for i, char in enumerate(sep_line):
        if char == '-' and sep_line[i-1] != '-':
            col_positions.append(i)
        if char != '-' and sep_line[i-1] == '-':
            col_positions.append(i)
    
    # Remove duplicates and sort
    col_positions = sorted(set(col_positions))
    
    # Extract column names from the header
    col_names = []
    for i in range(len(col_positions)-1):
        start = col_positions[i]
        end = col_positions[i+1]
        col_name = header_line[start:end].strip()
        col_names.append(col_name)
    # Add the last column
    col_names.append(header_line[col_positions[-1]:].strip())
    
    # Extract data rows
    data = []
    for line in lines[2:]:
        if line.strip() and not all(c == '-' for c in line):
            row_data = {}
            for i in range(len(col_positions)-1):
                start = col_positions[i]
                end = col_positions[i+1] if i+1 < len(col_positions) else len(line)
                value = line[start:end].strip()
                row_data[col_names[i]] = value
            # Add the last column if exists
            if len(col_positions) > 0:
                row_data[col_names[-1]] = line[col_positions[-1]:].strip()
            data.append(row_data)
    
    return pd.DataFrame(data)

def parse_ll1_table(result_text):
    """Extract and parse the LL(1) analysis table from the result text"""
    # Find the LL(1) table section
    ll1_match = re.search(r"TABLA DE ANÁLISIS LL\(1\)(.*?)(?=\n\n)", result_text, re.DOTALL)
    if not ll1_match:
        return None
    
    ll1_table_text = ll1_match.group(1)
    lines = ll1_table_text.strip().split('\n')
    
    # Skip separator lines
    data_lines = [line for line in lines if line.strip() and not all(c == '-' for c in line.strip())]
    
    if len(data_lines) < 2:
        return None
    
    # Parse the header row first
    header_line = data_lines[0].strip()
    terminals = []
    
    # Extract terminals from header, respecting fixed column width
    col_width = 20
    for i in range(1, len(header_line) // col_width + 1):
        start = i * col_width
        end = start + col_width
        if start < len(header_line):
            terminal = header_line[start:end].strip()
            if terminal:
                terminals.append(terminal)
    
    # Create DataFrame structure with non-terminals as rows and terminals as columns
    data = []
    
    # Process each row (non-terminal)
    for line in data_lines[1:]:
        row_data = {}
        
        # Extract non-terminal (first column)
        non_terminal = line[:col_width].strip()
        row_data["Non-Terminal"] = non_terminal
        
        # Extract productions for each terminal
        for i, terminal in enumerate(terminals):
            col_start = (i + 1) * col_width
            col_end = col_start + col_width
            
            if col_start < len(line):
                cell = line[col_start:col_end].strip()
                row_data[terminal] = cell
            else:
                row_data[terminal] = "-"
        
        data.append(row_data)
    
    return pd.DataFrame(data)

def style_dataframe(df, type_col=None):
    """Add styling to the dataframe for better visualization"""
    # Define CSS styles
    styles = [
        dict(selector="th", props=[("font-weight", "bold"), 
                                   ("background-color", "#4CAF50"),
                                   ("color", "white"),
                                   ("text-align", "center"),
                                   ("padding", "10px")]),
        dict(selector="td", props=[("padding", "8px"), 
                                   ("text-align", "center"),
                                   ("border", "1px solid #ddd")]),
        dict(selector="tr:nth-child(even)", props=[("background-color", "#f2f2f2")])
    ]
    
    # Apply styling
    styled_df = df.style.set_table_styles(styles).set_properties(**{
        'border': '1px solid #ddd',
        'text-align': 'center',
    })
    
    # Highlight non-terminals, terminals and special symbols if type column exists
    if type_col and type_col in df.columns:
        def highlight_row(row):
            # Get background color based on the type value
            if row[type_col] == 'V':
                bg_color = '#E3F2FD'  # Light blue for non-terminals
            elif row[type_col] == 'T':
                bg_color = '#F1F8E9'  # Light green for terminals
            else:
                bg_color = '#FFF3E0'  # Light orange for special symbols
                
            # Return a list with the same background color for all cells in the row
            return ['background-color: ' + bg_color] * len(row)
            
        styled_df = styled_df.apply(highlight_row, axis=1)
    
    return styled_df

def apply_enhanced_styling(df, highlight_column=None):
    """
    Apply enhanced styling to any dataframe with consistent colors
    highlight_column: optional column name to use for row-based coloring
    """
    # Base styling for all tables
    styled_df = df.style.set_properties(**{
        'background-color': '#f0f8ff',
        'color': 'black',
        'border': '1px solid #dddddd',
        'text-align': 'center',
        'font-size': '14px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#4CAF50'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '16px'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [('padding', '8px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
    ])
    
    # Apply row highlighting if a column is specified
    if highlight_column and highlight_column in df.columns:
        # Create a function that always returns a list of styles
        def highlight_row(row):
            colors = [''] * len(row)  # Initialize with empty strings
            
            # Default coloring for alternating rows
            row_style = 'background-color: #f0f8ff'
            
            # Customized color scheme based on value in the highlight column
            if highlight_column == "Tipo":
                value = row[highlight_column]
                if value == 'V':
                    row_style = 'background-color: #E3F2FD'  # Light blue for non-terminals
                elif value == 'T': 
                    row_style = 'background-color: #F1F8E9'  # Light green for terminals
                elif value == 'I':
                    row_style = 'background-color: #EDE7F6'  # Light purple for start symbol
                else:
                    row_style = 'background-color: #FFF3E0'  # Light orange for special symbols
            elif highlight_column == "ACCIÓN" or highlight_column == "Acción" or "accion" in highlight_column.lower():
                value = str(row[highlight_column])
                if "ERROR" in value:
                    row_style = 'background-color: #FFEBEE'  # Light red for errors
                elif "Match" in value:
                    row_style = 'background-color: #E8F5E9'  # Light green for matches
                elif "regla" in value.lower():
                    row_style = 'background-color: #E3F2FD'  # Light blue for rule applications
                else:
                    row_style = 'background-color: #F5F5F5'  # Light gray for other actions
            
            # Fill the entire row with the selected style
            return [row_style] * len(row)
        
        styled_df = styled_df.apply(highlight_row, axis=1)
    
    return styled_df

def style_symbol_table(df):
    """
    Apply custom styling to the symbol table with specific colors for each column:
    - Símbolo: keep the current color (blue for non-terminals)
    - Tipo: red when there's no production
    - FIRST and FOLLOW: green
    """
    # Base styling
    styled_df = df.style.set_properties(**{
        'background-color': '#f0f8ff',
        'color': 'black',
        'border': '1px solid #dddddd',
        'text-align': 'center',
        'font-size': '14px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#4CAF50'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '16px'),
            ('padding', '8px')
        ]},
        {'selector': 'td', 'props': [('padding', '8px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
    ])
    
    # Function to apply column-specific styling
    def highlight_columns(s):
        styles = [''] * len(s)
        
        for i, col in enumerate(df.columns):
            if col == 'Símbolo':
                # Apply color based on Type value in the same row
                if s['Tipo'] == 'V':
                    styles[i] = 'background-color: #E3F2FD'  # Light blue for non-terminals
                elif s['Tipo'] == 'T':
                    styles[i] = 'background-color: #F1F8E9'  # Light green for terminals
                elif s['Tipo'] == 'I':
                    styles[i] = 'background-color: #EDE7F6'  # Light purple for start symbol
                else:
                    styles[i] = 'background-color: #FFF3E0'  # Light orange for special symbols
            elif col == 'Tipo':
                # Red for Tipo when there's no production ("-" in FIRST or FOLLOW)
                if s['FIRST'] == '-' or s['FOLLOW'] == '-':
                    styles[i] = 'background-color: #FFEBEE'  # Light red
                else:
                    styles[i] = 'background-color: #F5F5F5'  # Light gray
            elif col in ['FIRST', 'FOLLOW']:
                styles[i] = 'background-color: #E8F5E9'  # Light green for FIRST and FOLLOW
        
        return styles
    
    # Apply the column-specific styling
    styled_df = styled_df.apply(highlight_columns, axis=1)
    
    return styled_df

def create_enhanced_ll1_table(df):
    """
    Create a better visualization for the LL(1) table with custom formatting similar to image #3
    """
    if df is None or df.empty:
        return None
    
    # Make a copy to avoid modifying the original
    formatted_df = df.copy()
    
    # Format the production cells to make them more readable
    for col in formatted_df.columns:
        if col == "Non-Terminal":
            continue
        formatted_df[col] = formatted_df[col].apply(
            lambda x: x.replace("->", "→") if isinstance(x, str) and "->" in x else x
        )
    
    # Apply styling
    styled_df = formatted_df.style.set_properties(**{
        'border': '1px solid #dddddd',
        'text-align': 'center',
        'font-size': '14px',
        'padding': '8px'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f2f2f2'),
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '14px'),
            ('padding', '8px'),
            ('border', '1px solid #dddddd')
        ]}
    ])
    
    # Define a function to style cells based on content
    def style_cells(val):
        if val == '-':
            return 'background-color: #ffcccc'  # Light red for empty cells
        elif '→' in str(val):
            return 'background-color: #ccffcc'  # Light green for production rules
        elif val == 'ε':
            return 'background-color: #ffe6cc'  # Light orange for epsilon
        else:
            return ''
    
    # Apply cell-by-cell styling
    styled_df = styled_df.applymap(style_cells)
    
    # Style the Non-Terminal column with light blue background
    for i in range(len(formatted_df)):
        styled_df.set_properties(subset=pd.IndexSlice[i, 'Non-Terminal'], **{
            'background-color': '#e6f2ff',
            'font-weight': 'bold'
        })
    
    return styled_df

def create_fallback_tree_visualization(parse_steps):
    """Create a proper hierarchical tree visualization based on the input string"""
    # Extract both production rules and token matches
    actions = []
    for step in parse_steps:
        action = step.get("accion", "")
        if action.startswith("Aplicar regla:"):
            rule = action.replace("Aplicar regla: ", "")
            left, right = rule.split("->")
            actions.append(("rule", left.strip(), right.strip()))
        elif action.startswith("Match:"):
            token = action.replace("Match: ", "")
            actions.append(("match", token))
    
    if not actions:
        st.warning("No hay suficientes datos para generar el árbol de derivación.")
        return
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Find the start symbol
    start_symbol = None
    for action_type, *args in actions:
        if action_type == "rule":
            start_symbol = args[0]
            break
            
    if not start_symbol:
        st.warning("No se pudo determinar el símbolo inicial.")
        return
    
    # Create tree structure with parent tracking
    node_counter = 0
    parent_stack = []
    terminal_nodes = []
    
    # Start with the root node
    root_id = f"node_{node_counter}"
    G.add_node(root_id, label=start_symbol, type="non-terminal")
    parent_stack.append((root_id, start_symbol))
    node_counter += 1
    
    # Process each action
    for action_type, *args in actions:
        if action_type == "rule":
            left, right = args
            
            # Find the parent node for this expansion
            parent_id = None
            parent_idx = -1
            for idx, (node_id, symbol) in enumerate(parent_stack):
                if symbol == left:
                    parent_id = node_id
                    parent_idx = idx
                    break
            
            if parent_id is not None:
                # Remove the expanded non-terminal from the stack
                parent_stack.pop(parent_idx)
                
                # Add child nodes
                right_symbols = right.split()
                if right == "ε":
                    right_symbols = ["ε"]
                    
                # Add nodes for each symbol
                for symbol in right_symbols:
                    child_id = f"node_{node_counter}"
                    node_counter += 1
                    
                    # Determine node type
                    node_type = "epsilon" if symbol == "ε" else \
                                "non-terminal" if (symbol[0].isupper() or "'" in symbol) else \
                                "terminal"
                    
                    # Add node and edge
                    G.add_node(child_id, label=symbol, type=node_type)
                    G.add_edge(parent_id, child_id)
                    
                    # If this is a non-terminal, add to stack for future processing
                    if node_type == "non-terminal":
                        parent_stack.append((child_id, symbol))
                    elif node_type == "terminal":
                        terminal_nodes.append(child_id)
    
    # Create a proper hierarchical tree layout
    plt.figure(figsize=(12, 8))
    plt.title("Árbol de Derivación", fontsize=16)
    
    try:
        # Use hierarchical layout
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        # Fallback to a custom layout function with better node separation
        pos = {}
        
        def assign_positions(node, level=0, x_pos=0, width=1.0, min_sep=1.5):
            """Position nodes in a hierarchical layout with minimum separation"""
            pos[node] = (x_pos, -level)  # Negative level for top-down layout
            children = list(G.successors(node))
            
            if not children:
                return x_pos, x_pos + width  # Return range taken by this subtree
            
            # Calculate total width needed for all children with separation
            child_width = width / len(children)
            if child_width < min_sep:
                child_width = min_sep
                
            # Position each child
            left_bound = x_pos - (child_width * (len(children) - 1) / 2)
            for i, child in enumerate(children):
                child_x = left_bound + i * child_width
                assign_positions(child, level + 1, child_x, child_width, min_sep)
            
            return x_pos - width/2, x_pos + width/2
        
        # Start positioning from the root
        roots = [n for n, d in G.in_degree() if d == 0]
        if roots:
            assign_positions(roots[0])
    
    # Draw nodes with different styles based on node type
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    
    # Use different colors for different node types
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'non-terminal':
            node_colors.append('lightblue')
        elif node_type == 'terminal':
            node_colors.append('lightgreen')
        elif node_type == 'epsilon':
            node_colors.append('lightyellow')
        else:
            node_colors.append('lightgray')
    
    # Draw the tree with larger node spacing
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors,
                          edgecolors='black', node_shape='o')
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrows=True, width=1.5, arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    
    st.caption("Árbol de derivación para la gramática y cadena de entrada analizada")

def main():
    # Setup session state to store editor content
    if 'grammar_text' not in st.session_state:
        st.session_state.grammar_text = """E -> T E'
E' -> + T E' | ε
T -> F T'
T' -> * F T' | ε
F -> ( E ) | id"""
    if 'input_text' not in st.session_state:
        st.session_state.input_text = "id + id * id"

    # Configuración de página con tema y estilo
    st.set_page_config(
        page_title="Analizador LL(1)",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Estilo CSS personalizado
    st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight:bold; color:#1E88E5; margin-bottom:20px}
    .medium-font {font-size:20px !important; font-weight:bold; color:#004D40; margin-top:15px; margin-bottom:10px}
    .result-header {font-size:22px; font-weight:bold; color:#0D47A1; border-bottom:2px solid #0D47A1; padding-bottom:8px; margin-top:20px}
    .success-text {color:#1B5E20; font-weight:bold}
    .error-text {color:#B71C1C; font-weight:bold}
    .highlight {background-color:#E3F2FD; padding:15px; border-radius:5px; margin:10px 0}
    .sidebar-title {font-size:24px; font-weight:bold; color:#283593; margin-bottom:20px}
    .sidebar-subtitle {font-size:18px; font-weight:bold; color:#455A64; margin-top:15px}
    button.keyboard-btn {margin: 2px; padding: 2px 8px;}
    </style>
    """, unsafe_allow_html=True)

    # Helper functions for virtual keyboard
    def add_to_grammar(symbol):
        st.session_state.grammar_text += symbol
    
    def add_to_input(symbol):
        st.session_state.input_text += symbol

    # Encabezado principal
    st.markdown('<p class="big-font">Analizador Sintáctico LL(1)</p>', unsafe_allow_html=True)
    st.markdown('Herramienta para análisis de gramáticas libres de contexto utilizando parsing LL(1)')
    
    # Sidebar con configuración
    st.sidebar.markdown('<p class="sidebar-title">Configuración</p>', unsafe_allow_html=True)
    
    # Grammar input options
    st.sidebar.markdown('<p class="sidebar-subtitle">Gramática</p>', unsafe_allow_html=True)
    input_method = st.sidebar.radio("Método de entrada:", ["Editor de texto", "Subir archivo"])
    
    grammar_text = ""
    if input_method == "Editor de texto":
        grammar_text = st.sidebar.text_area(
            "Ingrese su gramática:", 
            value=st.session_state.grammar_text, 
            height=200,
            key="grammar_editor",
            on_change=lambda: setattr(st.session_state, 'grammar_text', st.session_state.grammar_editor)
        )
        
        # Virtual keyboard for grammar symbols
        st.sidebar.markdown('<p class="sidebar-subtitle">Teclado virtual para gramática</p>', unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button("ε", on_click=add_to_grammar, args=["ε"], key="eps_btn", use_container_width=True)
            st.button("->", on_click=add_to_grammar, args=["->"], key="arrow_btn", use_container_width=True)
        with col2:
            st.button("|", on_click=add_to_grammar, args=[" | "], key="or_btn", use_container_width=True)
            st.button("id", on_click=add_to_grammar, args=["id"], key="id_btn", use_container_width=True)
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            st.button("+", on_click=add_to_grammar, args=[" + "], key="plus_btn", use_container_width=True)
            st.button("*", on_click=add_to_grammar, args=[" * "], key="star_btn", use_container_width=True)
        with col4:
            st.button("(", on_click=add_to_grammar, args=[" ( "], key="lparen_btn", use_container_width=True)
            st.button(")", on_click=add_to_grammar, args=[" ) "], key="rparen_btn", use_container_width=True)
    else:
        uploaded_grammar = st.sidebar.file_uploader("Subir archivo de gramática (.txt)", type=["txt"])
        if uploaded_grammar:
            grammar_text = uploaded_grammar.getvalue().decode("utf-8")
            st.session_state.grammar_text = grammar_text
            st.sidebar.success(f"Archivo '{uploaded_grammar.name}' cargado correctamente")
    
    # Grammar optimization options
    st.sidebar.markdown('<p class="sidebar-subtitle">Optimización de Gramática</p>', unsafe_allow_html=True)
    optimize_grammar = st.sidebar.checkbox("Optimizar gramática")
    
    if optimize_grammar:
        optimize_options = st.sidebar.multiselect(
            "Seleccione optimizaciones a aplicar:",
            ["Eliminar recursividad por izquierda", "Factorización por izquierda"],
            default=["Eliminar recursividad por izquierda"]
        )
    
    # Input string options
    st.sidebar.markdown('<p class="sidebar-subtitle">Cadena de entrada</p>', unsafe_allow_html=True)
    input_string_method = st.sidebar.radio("Método de entrada para cadena:", ["Ejemplos predefinidos", "Entrada manual", "Subir archivo"])
    
    input_text = ""
    if input_string_method == "Ejemplos predefinidos":
        input_examples = {
            "Ejemplo 1": "id + id * id",
            "Ejemplo 2": "( id )",
            "Ejemplo 3": "id + id + id",
            "Ejemplo 4": "id * id * id"
        }
        selected_example = st.sidebar.selectbox("Seleccione un ejemplo:", list(input_examples.keys()))
        input_text = input_examples[selected_example]
        st.session_state.input_text = input_text
    elif input_string_method == "Entrada manual":
        input_text = st.sidebar.text_area(
            "Ingrese la cadena de entrada:", 
            value=st.session_state.input_text, 
            height=80, 
            key="input_editor",
            on_change=lambda: setattr(st.session_state, 'input_text', st.session_state.input_editor)
        )
        
        # Virtual keyboard for input
        st.sidebar.markdown('<p class="sidebar-subtitle">Teclado virtual para entrada</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.button("id", on_click=add_to_input, args=["id "], key="input_id_btn", use_container_width=True)
        with col2:
            st.button("+", on_click=add_to_input, args=[" + "], key="input_plus_btn", use_container_width=True)
        with col3:
            st.button("*", on_click=add_to_input, args=[" * "], key="input_star_btn", use_container_width=True)
        
        col4, col5 = st.sidebar.columns(2)
        with col4:
            st.button("(", on_click=add_to_input, args=[" ( "], key="input_lparen_btn", use_container_width=True)
        with col5:
            st.button(")", on_click=add_to_input, args=[" ) "], key="input_rparen_btn", use_container_width=True)
    else:
        uploaded_input = st.sidebar.file_uploader("Subir archivo de entrada (.txt)", type=["txt"])
        if uploaded_input:
            input_text = uploaded_input.getvalue().decode("utf-8")
            st.session_state.input_text = input_text
            st.sidebar.success(f"Archivo '{uploaded_input.name}' cargado correctamente")
    
    # Contenido principal dividido en pestañas
    tab1, tab2, tab3 = st.tabs(["Entrada y Análisis", "Árbol de Derivación", "Ayuda"])
    
    with tab1:
        # Creamos dos columnas para organizar el contenido
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown('<p class="medium-font">Gramática Ingresada</p>', unsafe_allow_html=True)
            with st.container():
                if grammar_text:
                    st.code(grammar_text)
                else:
                    st.warning("Por favor ingrese una gramática o cargue un archivo")
            
            st.markdown('<p class="medium-font">Cadena de Entrada</p>', unsafe_allow_html=True)
            with st.container():
                if input_text:
                    st.code(input_text)
                else:
                    st.warning("Por favor ingrese una cadena de entrada o cargue un archivo")
            
            analyze_btn = st.button("Analizar Sintácticamente", 
                                  type="primary",
                                  disabled=not (grammar_text and input_text))
        
        with col2:
            if analyze_btn and grammar_text and input_text:
                try:
                    # Aplicar optimizaciones si están activadas
                    optimized_grammar = grammar_text
                    if optimize_grammar:
                        if "Eliminar recursividad por izquierda" in optimize_options:
                            optimized_grammar = eliminate_left_recursion(optimized_grammar)
                            st.success("✅ Recursividad por izquierda eliminada")
                        if "Factorización por izquierda" in optimize_options:
                            optimized_grammar = left_factorization(optimized_grammar)
                            st.success("✅ Factorización por izquierda aplicada")
                        
                        st.markdown('<p class="medium-font">Gramática Optimizada</p>', unsafe_allow_html=True)
                        st.code(optimized_grammar)
                    
                    # Crear archivos temporales para la gramática y la entrada
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                        safe_grammar = optimized_grammar.replace('ε', 'eps') if optimize_grammar else grammar_text.replace('ε', 'eps')
                        f.write(safe_grammar)
                        grammar_file = f.name
                        
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                        f.write(input_text)
                        input_file = f.name
                    
                    with st.spinner('Analizando gramática y cadena de entrada...'):
                        # Ejecutar análisis
                        result, parse_steps = parse_grammar_and_analyze(grammar_file, input_file, return_steps=True)
                    
                    st.markdown('<p class="result-header">Resultado del Análisis</p>', unsafe_allow_html=True)
                    
                    # Mostrar resultados en una tabla con estilo
                    if "RECHAZADA" in result:
                        st.error("❌ La cadena fue rechazada")
                    else:
                        st.success("✅ La cadena fue aceptada")
                    
                    # Parse the tables in a more visually appealing format
                    with st.expander("Ver resultado completo (texto plano)", expanded=False):
                        st.text(result)
                    
                    # Create tabs for each type of analysis table
                    results_tab1, results_tab2, results_tab3 = st.tabs(["Tabla de Símbolos", "Análisis de la Cadena", "Tabla de Análisis LL(1)"])
                    
                    # Tab 1: Display the Symbol Table (non-terminal symbols only)
                    with results_tab1:
                        symbol_df = parse_symbol_table(result)
                        if symbol_df is not None:
                            # Filter out terminal symbols and select only needed columns
                            non_terminals_df = symbol_df[symbol_df["Tipo"].isin(["I", "V"])][["Símbolo", "Tipo", "FIRST", "FOLLOW"]]
                            
                            if not non_terminals_df.empty:
                                try:
                                    st.dataframe(
                                        style_symbol_table(non_terminals_df),
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.warning(f"Error al aplicar estilos a la tabla: {str(e)}")
                                    st.dataframe(non_terminals_df, use_container_width=True)  # Fallback
                            else:
                                st.info("No se encontraron símbolos no terminales en la gramática.")
                        else:
                            st.warning("No se pudo extraer la tabla de símbolos del resultado.")
                    
                    # Tab 2: Display the Chain Analysis
                    with results_tab2:
                        chain_df = parse_chain_analysis(result)
                        if chain_df is not None:
                            try:
                                # Find the action column more safely
                                action_column = None
                                for col in chain_df.columns:
                                    if "ACCIÓN" in col or "ACCION" in col or "accion" in col.lower():
                                        action_column = col
                                        break
                                
                                if action_column:
                                    st.dataframe(apply_enhanced_styling(chain_df, action_column), use_container_width=True)
                                else:
                                    st.dataframe(apply_enhanced_styling(chain_df), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Error al aplicar estilos a la tabla: {str(e)}")
                                st.dataframe(chain_df, use_container_width=True)  # Fallback
                        else:
                            st.warning("No se pudo extraer el análisis de la cadena del resultado.")
                    
                    # Tab 3: Display the LL(1) Table with improved formatting
                    with results_tab3:
                        ll1_df = parse_ll1_table(result)
                        if ll1_df is not None:
                            try:
                                # Try to display with enhanced styling
                                st.dataframe(create_enhanced_ll1_table(ll1_df), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Error al aplicar estilos a la tabla: {str(e)}")
                                try:
                                    # Fall back to basic display
                                    st.table(ll1_df)
                                except:
                                    # Last resort: plain text display
                                    ll1_match = re.search(r"TABLA DE ANÁLISIS LL\(1\)(.*?)(?=\n\n)", result, re.DOTALL)
                                    if ll1_match:
                                        st.text(ll1_match.group(1))
                                    else:
                                        st.warning("No se pudo extraer la tabla de análisis LL(1) del resultado.")
                        else:
                            st.warning("No se pudo extraer la tabla de análisis LL(1) del resultado.")
                    
                    # Guardar los pasos de análisis para el árbol de derivación
                    if "ACEPTADA" in result:
                        st.session_state.parse_steps = parse_steps
                        st.session_state.input_text_analyzed = input_text
                        st.session_state.grammar_text_analyzed = grammar_text
                        st.info("👉 Vaya a la pestaña 'Árbol de Derivación' para ver la representación visual del análisis")
                    
                    # Limpiar archivos temporales
                    os.unlink(grammar_file)
                    os.unlink(input_file)
                
                except Exception as e:
                    st.error(f"Error en el análisis: {str(e)}")
            else:
                with st.container():
                    st.info("👈 Ingrese su gramática y cadena a analizar, luego presione el botón 'Analizar Sintácticamente'")
                    st.image("https://gramaticasformales.wordpress.com/wp-content/uploads/2010/12/6.png?w=640", 
                             caption="Ejemplo de Análisis Sintáctico LL(1)", width=400)
    
    with tab2:
        st.markdown('<p class="medium-font">Árbol de Derivación</p>', unsafe_allow_html=True)
        
        if 'parse_steps' in st.session_state and st.session_state.parse_steps:
            try:
                # Check if Graphviz is installed
                graphviz_installed = shutil.which('dot') is not None
                
                if graphviz_installed:
                    with st.spinner('Generando árbol de derivación con Graphviz...'):
                        try:
                            # Try to create the parse tree using Graphviz
                            dot = generate_parse_tree(st.session_state.parse_steps)
                            st.graphviz_chart(dot)
                            
                            # Provide download option for the tree
                            def get_graphviz_html():
                                return f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <title>Árbol de Derivación</title>
                                </head>
                                <body>
                                    {dot.pipe().decode('utf-8')}
                                </body>
                                </html>
                                """
                                
                            html_bytes = get_graphviz_html().encode()
                            b64 = base64.b64encode(html_bytes).decode()
                            
                            href = f'<a href="data:text/html;base64,{b64}" download="arbol_derivacion.html">Descargar Árbol HTML</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error al generar el árbol con Graphviz: {str(e)}")
                            st.warning("Usando visualización alternativa del árbol...")
                            # Fallback to alternative visualization
                            create_fallback_tree_visualization(st.session_state.parse_steps)
                else:
                    st.warning("⚠️ Graphviz no está instalado o no se encuentra en el PATH del sistema.")
                    st.info("Usando visualización alternativa del árbol...")
                    # Use fallback visualization method
                    create_fallback_tree_visualization(st.session_state.parse_steps)
                    
                    # Installation instruction for Graphviz
                    with st.expander("Instrucciones para instalar Graphviz"):
                        st.markdown("""
                        Para una mejor visualización del árbol de derivación, se recomienda instalar Graphviz:
                        
                        1. Descarga Graphviz desde [la página oficial](https://graphviz.org/download/)
                        2. Durante la instalación, asegúrate de seleccionar la opción "Añadir Graphviz al PATH del sistema"
                        3. Reinicia tu computadora después de la instalación
                        
                        Alternativamente, puedes instalarlo con:
                        - Windows: `winget install graphviz` o `choco install graphviz`
                        - macOS: `brew install graphviz`
                        - Linux: `sudo apt-get install graphviz` o equivalente en tu distribución
                        """)
            except Exception as e:
                st.error(f"Error al generar el árbol de derivación: {str(e)}")
        else:
            st.info("Realice un análisis exitoso en la pestaña 'Entrada y Análisis' para visualizar el árbol de derivación.")
    
    with tab3:
        st.markdown('<p class="medium-font">Guía del Analizador LL(1)</p>', unsafe_allow_html=True)
        with st.expander("¿Cómo ingresar una gramática?", expanded=True):
            st.markdown("""
            - Cada regla debe estar en una línea separada
            - Usa `->` para separar los no terminales de sus producciones
            - Usa `|` para separar producciones alternativas
            - Usa `ε` o `eps` para representar la cadena vacía
            - Ejemplo: `A -> a B | ε`
            """)
        
        with st.expander("Ejemplos de gramáticas"):
            st.code("""
S -> id | S + S | S * S | ( S )

E -> T E'
E' -> + T E' | ε 
T -> F T'
T' -> * F T' | ε
F -> ( E ) | id

A -> B a | C d
B -> b B | ε
C -> c C | ε
            """)
        
        with st.expander("Optimización de gramáticas"):
            st.markdown("""
            **Eliminación de recursividad por izquierda:**
            - Transforma reglas recursivas por izquierda en reglas equivalentes no recursivas
            - Ejemplo: `A -> A α | β` se convierte en:
              ```
              A -> β A'
              A' -> α A' | ε
              ```
            
            **Factorización por izquierda:**
            - Refactoriza reglas con prefijos comunes para facilitar decisiones deterministas
            - Ejemplo: `A -> α β | α γ` se convierte en:
              ```
              A -> α A'
              A' -> β | γ
              ```
            """)
        
        with st.expander("Sobre el análisis LL(1)"):
            st.markdown("""
            El análisis LL(1) es un método de análisis sintáctico predictivo que construye el árbol 
            de derivación empezando desde la raíz (de arriba hacia abajo) y de izquierda a derecha.
            
            El '1' en LL(1) indica que solo se necesita 1 símbolo de anticipación para tomar decisiones.
            
            Este analizador implementa:
            - Cálculo de conjuntos FIRST y FOLLOW
            - Construcción de tabla de análisis LL(1)
            - Análisis de cadenas mediante algoritmo predictivo
            - Visualización del árbol de derivación
            """)

if __name__ == "__main__":
    main()