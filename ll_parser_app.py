import streamlit as st
import tempfile
import os
import base64
import shutil
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import re
import csv
from LL_parser import parse_grammar_and_analyze, eliminate_left_recursion, left_factorization, generate_parse_tree
from typing import List, Dict, Any, Optional, Tuple
import streamlit.components.v1 as components

# Helper functions to read CSV tables
def read_symbol_table_csv(csv_file_path):
    """Read the symbol table from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        # Filter out non-meaningful rows
        df = df[df['Símbolo'].notna()]
        return df
    except Exception as e:
        st.warning(f"Error reading symbol table CSV: {str(e)}")
        return None

def read_ll1_table_csv(csv_file_path):
    """Read the LL(1) analysis table from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        # Filter out empty rows
        df = df[df['Non-Terminal'].notna()]
        return df
    except Exception as e:
        st.warning(f"Error reading LL(1) table CSV: {str(e)}")
        return None

def read_chain_analysis_csv(csv_file_path):
    """Read the chain analysis from a CSV file"""
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        st.warning(f"Error reading chain analysis CSV: {str(e)}")
        return None

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
        # Use a more reliable way to extract columns based on fixed positions
        if len(line) < 15:  # Skip short lines
            continue
            
        # Extract columns based on positions observed in the formatted output
        try:
            simbolo = line[:15].strip()
            tipo = line[15:25].strip()
            nullable = line[25:37].strip()
            
            # For FIRST and FOLLOW, extract everything and look for the braces pattern
            rest_of_line = line[37:].strip()
            
            # Find FIRST (inside curly braces)
            first_match = re.search(r'(\{[^}]*\}|-)', rest_of_line)
            if first_match:
                first = first_match.group(0)
                # Extract FOLLOW after FIRST
                follow_match = re.search(r'(\{[^}]*\}|-)', rest_of_line[first_match.end():])
                if follow_match:
                    follow = follow_match.group(0)
                else:
                    follow = "-"
            else:
                first = "-"
                follow = "-"
                
            data.append({
                "Símbolo": simbolo,
                "Tipo": tipo,
                "NULLABLE": nullable,
                "FIRST": first,
                "FOLLOW": follow
            })
        except Exception:
            # If extraction fails, try the original splitting approach
            parts = [p for p in re.split(r'\s{2,}', line.strip()) if p]
            if len(parts) >= 4:
                data.append({
                    "Símbolo": parts[0],
                    "Tipo": parts[1],
                    "NULLABLE": parts[2] if len(parts) > 2 else "-",
                    "FIRST": parts[3] if len(parts) > 3 else "-",
                    "FOLLOW": parts[4] if len(parts) > 4 else "-"
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
    
    # More reliable column detection based on fixed width
    col_width = 20  # Each column is 20 chars wide as specified in LL_parser.py
    
    # Extract terminals from header based on fixed column width
    terminals = []
    first_col = header_line[:col_width].strip()  # First column is "Non-Terminal"
    
    # Extract remaining headers in fixed-width chunks
    for i in range(1, (len(header_line) // col_width) + 1):
        start = i * col_width
        end = min(start + col_width, len(header_line))
        terminal = header_line[start:end].strip()
        if terminal:
            terminals.append(terminal)
    
    # Create DataFrame structure with non-terminals as rows and terminals as columns
    data = []
    
    # Process each row (non-terminal)
    for line in data_lines[1:]:
        if not line.strip() or all(c == '-' for c in line) or len(line) < col_width:
            continue
            
        row_data = {}
        
        # Extract non-terminal (first column)
        non_terminal = line[:col_width].strip()
        row_data["Non-Terminal"] = non_terminal
        
        # Extract productions for each terminal based on fixed width
        for i, terminal in enumerate(terminals):
            col_idx = i + 1  # Skip the first column (non-terminal)
            start = col_idx * col_width
            end = min(start + col_width, len(line))
            
            if start < len(line):
                cell = line[start:end].strip()
                row_data[terminal] = cell if cell else "-"
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

def create_fallback_tree_visualization(parse_steps: List[Dict[str, Any]]):
    """
    Crea una visualización del árbol de derivación jerárquica usando NetworkX y Matplotlib.
    Sigue las transiciones exactas del análisis de la cadena.
    """
    # Extraer reglas (excluyendo epsilons)
    rules = []
    for step in parse_steps:
        action = step.get("accion", "")
        if action.startswith("Aplicar regla:"):
            rule = action.replace("Aplicar regla: ", "")
            try:
                left, right = rule.split("->")
                # No omitimos reglas epsilon aquí para mantener la secuencia correcta
                rules.append((left.strip(), right.strip()))
            except ValueError:
                continue
    
    if not rules:
        st.warning("No hay suficientes reglas para generar el árbol de derivación.")
        return
    
    # Crear un grafo dirigido (pero lo construiremos como un árbol)
    G = nx.DiGraph()
    
    # Variables para el seguimiento del árbol
    root_symbol = rules[0][0]  # La primera regla indica el símbolo inicial
    node_counter = 0
    node_symbols = {}  # Mapeo de ID a símbolo
    node_types = {}    # Mapeo de ID a tipo (terminal/no-terminal)
    node_expansions = {}  # Mapeo de nodos no terminales a su estado de expansión
    
    # Crear nodo raíz
    root_id = f"node_{node_counter}"
    G.add_node(root_id)
    node_symbols[root_id] = root_symbol
    node_types[root_id] = "non-terminal"
    node_expansions[root_id] = False  # Aún no expandido
    node_counter += 1
    
    # Seguimiento de la forma sentencial actual como lista de IDs de nodos
    sentential_form = [root_id]
    
    # Procesar cada regla en el orden de aplicación
    for left, right in rules:
        # Encontrar el no terminal más a la izquierda sin expandir que coincida con 'left'
        expand_index = None
        for i, node_id in enumerate(sentential_form):
            if (node_id in node_symbols and 
                node_symbols[node_id] == left and 
                node_id in node_expansions and 
                not node_expansions[node_id]):
                expand_index = i
                break
        
        if expand_index is None:
            # No se encontró un no terminal sin expandir que coincida, omitir esta regla
            continue
        
        # Marcar este no terminal como expandido
        current_node_id = sentential_form[expand_index]
        node_expansions[current_node_id] = True
        
        # Si es epsilon, simplemente marcamos el nodo como expandido pero no agregamos hijos
        # (No mostramos el epsilon en el árbol)
        if right.strip() in ["ε", "eps"]:
            continue
        
        # Manejar el lado derecho de la regla
        right_symbols = right.split()
        new_node_ids = []
        
        # Procesar cada símbolo en el lado derecho de la regla
        for symbol in right_symbols:
            new_id = f"node_{node_counter}"
            node_counter += 1
            G.add_node(new_id)
            G.add_edge(current_node_id, new_id)
            node_symbols[new_id] = symbol
            
            # Determinar si el símbolo es terminal o no terminal
            is_terminal = not (symbol[0].isupper() or "'" in symbol)
            node_types[new_id] = "terminal" if is_terminal else "non-terminal"
            
            if not is_terminal:
                # Agregar al seguimiento de expansión
                node_expansions[new_id] = False
            
            new_node_ids.append(new_id)
        
        # Actualizar la forma sentencial - reemplazar el no terminal expandido con su expansión
        sentential_form = sentential_form[:expand_index] + new_node_ids + sentential_form[expand_index+1:]
    
    # Crear layout jerárquico
    plt.figure(figsize=(12, 8))
    plt.title("Árbol de Derivación", fontsize=16)
    
    try:
        # Intentar usar el algoritmo de graphviz a través de pygraphviz
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        # Función para crear un layout jerárquico personalizado
        def hierarchical_pos(G, root=None, width=1., vert_gap=0.4, vert_loc=0, xcenter=0.5):
            """
            Crea un layout jerárquico posicionando los nodos por nivel.
            """
            if root is None:
                # Encontrar la raíz - nodo con grado de entrada 0
                roots = [n for n, d in G.in_degree() if d == 0]
                if roots:
                    root = roots[0]
                else:
                    # Si no hay raíz, usar el primer nodo
                    root = list(G.nodes())[0]
            
            def _hierarchy_pos(G, root, width=1., vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None):
                if pos is None:
                    pos = {root: (xcenter, vert_loc)}
                else:
                    pos[root] = (xcenter, vert_loc)
                
                children = list(G.neighbors(root))
                if not children:
                    return pos
                
                # Distribuir los hijos uniformemente
                dx = width / len(children)
                nextx = xcenter - width/2 - dx/2
                
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                        vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
                return pos
            
            return _hierarchy_pos(G, root, width=width, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=xcenter)
        
        pos = hierarchical_pos(G)
    
    # Preparar los atributos visuales
    node_colors = []
    for node in G.nodes():
        if node_types[node] == "terminal":
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightblue')
    
    # Dibujar los nodos con etiquetas (símbolos)
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=node_symbols, font_weight='bold', font_size=11)
    nx.draw_networkx_edges(G, pos, arrows=False, width=1.5)
    
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    
    st.caption("Árbol de derivación para la gramática y cadena de entrada analizada")

def check_grammar_issues(grammar_text):
    """
    Check if a grammar has left recursion or needs left factorization
    Returns a tuple (has_left_recursion, needs_factorization)
    """
    has_left_recursion = False
    needs_factorization = False
    
    lines = grammar_text.strip().split('\n')
    rules = {}
    
    # Parse the grammar
    for line in lines:
        if '->' not in line:
            continue
        left, right = line.split('->', 1)
        non_terminal = left.strip()
        productions = [p.strip() for p in right.split('|')]
        if non_terminal not in rules:
            rules[non_terminal] = []
        rules[non_terminal].extend(productions)
    
    # Check for left recursion
    for A, productions in rules.items():
        for prod in productions:
            prod_parts = prod.split()
            if prod_parts and prod_parts[0] == A:
                has_left_recursion = True
                break
        if has_left_recursion:
            break
    
    # Check for need of left factorization
    for A, productions in rules.items():
        prefixes = {}
        for prod in productions:
            prod_parts = prod.split()
            if not prod_parts:
                continue
                
            prefix = prod_parts[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(prod)
        
        # If multiple productions share the same prefix, factorization is needed
        for prefix, prods in prefixes.items():
            if len(prods) > 1:
                needs_factorization = True
                break
        if needs_factorization:
            break
    
    return has_left_recursion, needs_factorization

def main():
    # Setup session state to store editor content
    if 'grammar_text' not in st.session_state:
        st.session_state.grammar_text = """Struct -> struct Nombre { Comps }
Nombre -> id
Comps -> Comp Comps'
Comps' -> ; Comp Comps' | ε
Comp -> Type id
Type -> Typep | struct id | Pointer
Typep -> int | char | bool | float
Pointer -> * id"""
    if 'input_text' not in st.session_state:
        st.session_state.input_text = "struct id { int id ; struct id id ; * id id }"
    if 'optimization_grammar' not in st.session_state:
        st.session_state.optimization_grammar = """S -> S + S | S * S | id"""

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
    
    # Helper function for virtual keyboard - simplify to append at the end
    def add_to_grammar(symbol):
        st.session_state.grammar_text += symbol

    def add_to_optimization_grammar(symbol):
        st.session_state.optimization_grammar += symbol

    # Add helper function for grammar download buttons
    def add_grammar_download(original, optimized, filename="gramatica_optimizada.txt"):
        """Add a download button for the optimized grammar if it's different from the original"""
        if original.strip() != optimized.strip():
            st.download_button(
                label="📥 Descargar gramática optimizada",
                data=optimized,
                file_name=filename,
                mime="text/plain",
                help="Descargar la versión optimizada de la gramática"
            )

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
        
        # Virtual keyboard for grammar symbols - without + and *
        st.sidebar.markdown('<p class="sidebar-subtitle">Teclado virtual para gramática</p>', unsafe_allow_html=True)
        st.sidebar.caption("Los símbolos se agregarán al final del texto")
        
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.button("ε", on_click=add_to_grammar, args=["ε"], key="eps_btn", use_container_width=True)
        with col2:
            st.button("->", on_click=add_to_grammar, args=["->"], key="arrow_btn", use_container_width=True)
        with col3:
            st.button("|", on_click=add_to_grammar, args=[" | "], key="or_btn", use_container_width=True)
        
        col4, col5, col6 = st.sidebar.columns(3)
        with col4:
            st.button("id", on_click=add_to_grammar, args=["id"], key="id_btn", use_container_width=True)
        with col5:
            st.button("(", on_click=add_to_grammar, args=[" ( "], key="lparen_btn", use_container_width=True)
        with col6:
            st.button(")", on_click=add_to_grammar, args=[" ) "], key="rparen_btn", use_container_width=True)
    else:
        uploaded_grammar = st.sidebar.file_uploader("Subir archivo de gramática (.txt)", type=["txt"])
        if uploaded_grammar:
            grammar_text = uploaded_grammar.getvalue().decode("utf-8")
            st.session_state.grammar_text = grammar_text
            st.sidebar.success(f"Archivo '{uploaded_grammar.name}' cargado correctamente")
    
    # Input string options - simplified to just manual and file upload
    st.sidebar.markdown('<p class="sidebar-subtitle">Cadena de entrada</p>', unsafe_allow_html=True)
    input_string_method = st.sidebar.radio("Método de entrada para cadena:", ["Entrada manual", "Subir archivo"])
    
    input_text = ""
    if input_string_method == "Entrada manual":
        input_text = st.sidebar.text_area(
            "Ingrese la cadena de entrada:", 
            value=st.session_state.input_text, 
            height=80, 
            key="input_editor",
            on_change=lambda: setattr(st.session_state, 'input_text', st.session_state.input_editor)
        )
    else:
        uploaded_input = st.sidebar.file_uploader("Subir archivo de entrada (.txt)", type=["txt"])
        if uploaded_input:
            input_text = uploaded_input.getvalue().decode("utf-8")
            st.session_state.input_text = input_text
            st.sidebar.success(f"Archivo '{uploaded_input.name}' cargado correctamente")
    
    # Reorder tabs - Árbol de Derivación comes before Optimizar Gramática
    tab1, tab3, tab2, tab4 = st.tabs(["Entrada y Análisis", "Árbol de Derivación", "Optimizar Gramática", "Ayuda"])
    
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
                    # Check if grammar has issues before analyzing
                    has_left_recursion, needs_factorization = check_grammar_issues(grammar_text)
                    
                    # Instead of returning, use a flag to control the flow
                    should_continue_analysis = True
                    
                    if has_left_recursion or needs_factorization:
                        should_continue_analysis = False
                        st.warning("⚠️ La gramática ingresada no es compatible con LL(1):")
                        if has_left_recursion:
                            st.warning("• Se detectó recursividad por izquierda")
                        if needs_factorization:
                            st.warning("• Se detectó la necesidad de factorización por izquierda")
                        
                        # Copy the grammar to the optimization tab for convenience
                        st.session_state.optimization_grammar = grammar_text
                        
                        st.error("❌ No se puede analizar una cadena con una gramática que no es LL(1)")
                        st.info("👉 Vaya a la pestaña 'Optimizar Gramática' para transformar su gramática a una forma compatible con LL(1)")
                    
                    if should_continue_analysis:
                        # Proceed with analysis only if grammar is LL(1) compatible
                        # Crear archivos temporales para la gramática y la entrada
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                            safe_grammar = grammar_text.replace('ε', 'eps')
                            f.write(safe_grammar)
                            grammar_file = f.name
                            
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                            f.write(input_text)
                            input_file = f.name
                        
                        with st.spinner('Analizando gramática y cadena de entrada...'):
                            # Ejecutar análisis
                            result, parse_steps, csv_files = parse_grammar_and_analyze(grammar_file, input_file, return_steps=True)
                            
                            # Unpack CSV file paths
                            symbol_table_csv, ll1_table_csv, analysis_table_csv = csv_files
                        
                        st.markdown('<p class="result-header">Resultado del Análisis</p>', unsafe_allow_html=True)
                        
                        # Display analysis failure specific messages
                        if "RECHAZADA" in result:
                            st.error("❌ La cadena fue rechazada")
                            
                            # Check for specific error patterns
                            if "No hay producción para" in result:
                                st.error("Error de análisis LL(1): No se encontró una producción adecuada en la tabla.")
                                if has_left_recursion or needs_factorization:
                                    st.info("Esto puede deberse a que la gramática no está en formato LL(1). Consulte la pestaña 'Optimizar Gramática'.")
                        else:
                            st.success("✅ La cadena fue aceptada")
                        
                        # Create tabs for each type of analysis table
                        results_tab1, results_tab2, results_tab3 = st.tabs(["Tabla de Símbolos", "Análisis de la Cadena", "Tabla de Análisis LL(1)"])
                        
                        # Tab 1: Display the Symbol Table (non-terminal symbols only)
                        with results_tab1:
                            # Try to read from CSV first
                            symbol_df = read_symbol_table_csv(symbol_table_csv)
                            
                            # If CSV reading fails, fall back to text parsing
                            if symbol_df is None:
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
                            # Try to read from CSV first
                            chain_df = read_chain_analysis_csv(analysis_table_csv)
                            
                            # If CSV reading fails, fall back to text parsing
                            if chain_df is None:
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
                            # Try to read from CSV first
                            ll1_df = read_ll1_table_csv(ll1_table_csv)
                            
                            # If CSV reading fails, fall back to text parsing
                            if ll1_df is None:
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
                        
                        # Add download button at the end
                        st.download_button(
                            label="📥 Descargar resultado completo",
                            data=result,
                            file_name="analisis_ll1.txt",
                            mime="text/plain",
                            help="Descargar el resultado completo del análisis"
                        )
                        
                        # Limpiar archivos temporales
                        os.unlink(grammar_file)
                        os.unlink(input_file)
                        os.unlink(symbol_table_csv)
                        os.unlink(ll1_table_csv)
                        os.unlink(analysis_table_csv)
                    
                except Exception as e:
                    st.error(f"Error en el análisis: {str(e)}")
                    # Suggest optimization if there's an error during analysis
                    st.info("Si la gramática no está en formato LL(1), pruebe la pestaña 'Optimizar Gramática'")
            else:
                with st.container():
                    st.info("👈 Ingrese su gramática y cadena a analizar, luego presione el botón 'Analizar Sintácticamente'")
                    st.image("https://gramaticasformales.wordpress.com/wp-content/uploads/2010/12/6.png?w=640", 
                             caption="Ejemplo de Análisis Sintáctico LL(1)", width=400)
    
    with tab3:
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
                            # Fallback to alternative visualization
                            create_fallback_tree_visualization(st.session_state.parse_steps)
                else:
                    # Use fallback visualization method
                    create_fallback_tree_visualization(st.session_state.parse_steps)
                    
            except Exception as e:
                st.error(f"Error al generar el árbol de derivación: {str(e)}")
        else:
            st.info("Realice un análisis exitoso en la pestaña 'Entrada y Análisis' para visualizar el árbol de derivación.")
    
    with tab2:
        st.markdown('<p class="medium-font">Optimizar Gramática</p>', unsafe_allow_html=True)
        
        # Create two columns for the optimizations
        opt_col1, opt_col2 = st.columns([1, 1.5])
        
        with opt_col1:
            st.markdown("### Entrada de Gramática")
            
            opt_input_method = st.radio("Método de entrada:", ["Editor de texto", "Subir archivo"], key="opt_input_method")
            
            optimization_grammar = ""
            if opt_input_method == "Editor de texto":
                optimization_grammar = st.text_area(
                    "Ingrese su gramática:", 
                    value=st.session_state.optimization_grammar, 
                    height=200,
                    key="optimization_grammar_editor",
                    on_change=lambda: setattr(st.session_state, 'optimization_grammar', st.session_state.optimization_grammar_editor)
                )
                
                # Add virtual keyboard for grammar symbols
                st.caption("Teclado virtual para gramática")
                
                opt_col1_1, opt_col1_2, opt_col1_3 = st.columns(3)
                with opt_col1_1:
                    st.button("ε", on_click=add_to_optimization_grammar, args=["ε"], key="opt_eps_btn", use_container_width=True)
                with opt_col1_2:
                    st.button("->", on_click=add_to_optimization_grammar, args=["->"], key="opt_arrow_btn", use_container_width=True)
                with opt_col1_3:
                    st.button("|", on_click=add_to_optimization_grammar, args=[" | "], key="opt_or_btn", use_container_width=True)
            else:
                uploaded_opt_grammar = st.file_uploader("Subir archivo de gramática (.txt)", type=["txt"], key="uploaded_opt_grammar")
                if uploaded_opt_grammar:
                    optimization_grammar = uploaded_opt_grammar.getvalue().decode("utf-8")
                    st.session_state.optimization_grammar = optimization_grammar
                    st.success(f"Archivo '{uploaded_opt_grammar.name}' cargado correctamente")
            
            analyze_grammar_btn = st.button("Analizar Gramática", 
                                           type="primary",
                                           disabled=not optimization_grammar,
                                           key="analyze_grammar_btn")
        
        with opt_col2:
            if analyze_grammar_btn and optimization_grammar:
                st.markdown("### Resultados del Análisis")
                
                # Check if grammar has issues
                has_left_recursion, needs_factorization = check_grammar_issues(optimization_grammar)
                
                st.markdown("#### Diagnóstico de la Gramática")
                
                is_ll1_compatible = not (has_left_recursion or needs_factorization)
                
                if is_ll1_compatible:
                    st.success("✅ La gramática es compatible con LL(1)")
                else:
                    st.warning("⚠️ La gramática necesita optimización para ser compatible con LL(1)")
                
                col_lr, col_lf = st.columns(2)
                
                with col_lr:
                    if has_left_recursion:
                        st.warning("⚠️ Se detectó recursividad por izquierda")
                    else:
                        st.success("✅ Sin recursividad por izquierda")
                
                with col_lf:
                    if needs_factorization:
                        st.warning("⚠️ Necesita factorización por izquierda")
                    else:
                        st.success("✅ No requiere factorización")
                
                if not is_ll1_compatible:
                    st.markdown("#### Gramática Optimizada")
                    
                    # Apply transformations
                    optimized_grammar = optimization_grammar
                    
                    if has_left_recursion:
                        lr_grammar = eliminate_left_recursion(optimized_grammar)
                        with st.expander("Gramática sin recursividad por izquierda", expanded=True):
                            st.code(lr_grammar)
                            # Add download button only if grammar was actually changed
                            add_grammar_download(optimized_grammar, lr_grammar, "gramatica_sin_recursion.txt")
                        optimized_grammar = lr_grammar
                    
                    if needs_factorization:
                        lf_grammar = left_factorization(optimized_grammar)
                        with st.expander("Gramática con factorización por izquierda", expanded=True):
                            st.code(lf_grammar)
                            # Add download button only if grammar was actually changed
                            add_grammar_download(optimized_grammar, lf_grammar, "gramatica_factorizada.txt")
                        optimized_grammar = lf_grammar
                    
                    # Final optimized grammar if both transformations were applied
                    if has_left_recursion and needs_factorization:
                        with st.expander("Gramática completamente optimizada para LL(1)", expanded=True):
                            st.code(optimized_grammar)
                            # Add download button for final optimized grammar
                            add_grammar_download(optimization_grammar, optimized_grammar, "gramatica_ll1_completa.txt")
                    
                    st.info("Para usar esta gramática, cópiela y péguela en la pestaña 'Entrada y Análisis'")
                
                # No need for "siguiente paso" info if grammar is already LL(1)
                if is_ll1_compatible:
                    st.markdown("#### ¿Siguiente paso?")
                    st.success("La gramática ya es compatible con LL(1). Puede usarla directamente en la pestaña de Análisis.")
            else:
                st.info("👈 Ingrese una gramática para analizarla y optimizarla automáticamente")
                
                st.markdown("### ¿Por qué optimizar?")
                st.markdown("""
                Un analizador LL(1) requiere gramáticas específicas sin:
                
                1. **Recursividad por izquierda**: Reglas donde un no terminal deriva a sí mismo como primer símbolo
                   - Ejemplo: `A -> A α | β`
                   
                2. **Ambigüedad de prefijos**: Reglas con producciones alternativas que comparten prefijos
                   - Ejemplo: `A -> α β | α γ`
                
                Esta herramienta transforma automáticamente tu gramática para hacerla compatible con LL(1).
                """)
                
                with st.expander("Ver ejemplo de optimización"):
                    st.code("""
# Gramática original (con recursividad por izquierda)
E -> E + T | T

# Gramática optimizada (sin recursividad por izquierda)
E -> T E'
E' -> + T E' | ε
                    """)
    
    with tab4:
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

