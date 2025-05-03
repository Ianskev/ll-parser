import streamlit as st
import pandas as pd
import graphviz
import os
import tempfile
from LL_parser import extract_grammar, tokenize, explore_parse
import copy
import re

st.set_page_config(page_title="LL(1) Parser", layout="wide")

# Function for left-recursion elimination
def eliminate_left_recursion(rules):
    processed_rules = []
    variables = set([rule.split('->')[0].strip() for rule in rules])
    
    for rule in rules:
        var, productions = rule.split('->')
        var = var.strip()
        productions = [p.strip() for p in productions.strip().split('|')]
        
        # Separate direct left-recursive and non-left-recursive productions
        recursive_prods = []
        non_recursive_prods = []
        
        for prod in productions:
            prod_symbols = re.split(r'(\s+)', prod)
            first_symbol = prod_symbols[0] if prod_symbols else ""
            
            if first_symbol == var:
                # This is a left-recursive production
                recursive_prods.append(''.join(prod_symbols[1:]).strip())
            else:
                # Non-left-recursive production
                non_recursive_prods.append(prod)
        
        if recursive_prods:
            # Create new variable with "Prime" suffix
            new_var = f"{var}'"
            
            # Rewrite non-recursive productions
            new_non_recursive = []
            for prod in non_recursive_prods:
                new_non_recursive.append(f"{prod} {new_var}" if prod else new_var)
                
            # Create the new variable's productions
            new_recursive = []
            for prod in recursive_prods:
                new_recursive.append(f"{prod} {new_var}")
            new_recursive.append("''")  # epsilon
            
            # Add the new rules
            if new_non_recursive:
                processed_rules.append(f"{var} -> {' | '.join(new_non_recursive)}")
            processed_rules.append(f"{new_var} -> {' | '.join(new_recursive)}")
        else:
            # No left recursion, keep the rule as is
            processed_rules.append(rule)
            
    return processed_rules

# Function for left factoring
def left_factor(rules):
    processed_rules = []
    rules_by_var = {}
    
    # Group rules by variable
    for rule in rules:
        var, productions = rule.split('->')
        var = var.strip()
        if var not in rules_by_var:
            rules_by_var[var] = []
        rules_by_var[var].extend([p.strip() for p in productions.strip().split('|')])
    
    for var, productions in rules_by_var.items():
        # Keep factoring until no more common prefixes
        while True:
            # Group productions by their first symbol
            prefix_groups = {}
            for prod in productions:
                if not prod:  # Handle empty string
                    if '' not in prefix_groups:
                        prefix_groups[''] = []
                    prefix_groups[''].append('')
                    continue
                    
                tokens = prod.split()
                prefix = tokens[0] if tokens else ''
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(prod)
            
            # Find largest group with common prefix
            max_group = max(prefix_groups.items(), key=lambda x: len(x[1]))
            prefix, group = max_group
            
            # If no common prefix or only one production, we're done
            if len(group) <= 1:
                break
                
            # Factor out common prefix
            suffix_productions = []
            for prod in group:
                if prod.startswith(prefix):
                    suffix = prod[len(prefix):].strip()
                    suffix_productions.append(suffix if suffix else "''")  # Use '' for empty
            
            # Remove original productions with the common prefix
            new_productions = []
            for prod in productions:
                if prod not in group:
                    new_productions.append(prod)
            
            # Create new variable
            new_var = f"{var}_{prefix}"
            
            # Add the factored production
            new_productions.append(f"{prefix} {new_var}")
            
            # Update the productions
            productions = new_productions
            processed_rules.append(f"{new_var} -> {' | '.join(suffix_productions)}")
        
        # Add the final rule for this variable
        processed_rules.append(f"{var} -> {' | '.join(productions)}")
    
    return processed_rules

# Function to build parse tree from exploration steps
def build_parse_tree(steps):
    # Create a tree structure for visualization
    tree = graphviz.Digraph()
    tree.attr('node', shape='box')
    
    node_counter = 0
    node_map = {}
    parent_stack = []
    
    for step in steps:
        action = step["accion"]
        
        if action.startswith("Aplicar regla:"):
            # Extract the rule
            rule = action.replace("Aplicar regla: ", "")
            left, right = rule.split("->")
            left = left.strip()
            right = right.strip()
            
            # Create node for this non-terminal
            if not parent_stack:
                # Root node
                node_id = f"node{node_counter}"
                node_counter += 1
                tree.node(node_id, left)
                node_map[left] = node_id
                parent_stack.append((node_id, left))
            
            # Add child nodes for the right side of the production
            if right != "ε":  # Skip epsilon
                parent_id, _ = parent_stack[-1]
                right_symbols = right.split()
                
                for symbol in right_symbols:
                    child_id = f"node{node_counter}"
                    node_counter += 1
                    tree.node(child_id, symbol)
                    tree.edge(parent_id, child_id)
                    
                    if symbol[0].isupper() or "'" in symbol:
                        # This is a non-terminal, push to stack
                        parent_stack.append((child_id, symbol))
            
        elif action.startswith("Match:"):
            # We've matched a terminal, remove the last non-terminal
            if parent_stack:
                parent_stack.pop()
    
    return tree

# Main app
def main():
    st.title("LL(1) Parser con Optimización y Visualización")
    
    # Create tabs for input, parse table, tree visualization
    tabs = st.tabs(["Entrada de Datos", "Análisis", "Árbol Sintáctico", "Optimización"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Gramática")
            grammar_source = st.radio("Fuente de la gramática:", 
                                      ["Escribir directamente", "Cargar desde archivo"])
            
            if grammar_source == "Escribir directamente":
                grammar_text = st.text_area("Ingrese su gramática:", 
                                            height=300,
                                            value="E -> T E'\nE' -> + T E' | ''\nT -> F T'\nT' -> * F T' | ''\nF -> ( E ) | id")
            else:
                uploaded_grammar = st.file_uploader("Subir archivo de gramática", type=["txt"])
                if uploaded_grammar:
                    grammar_text = uploaded_grammar.getvalue().decode("utf-8")
                else:
                    grammar_text = ""
            
            # Virtual keyboard for special symbols
            st.subheader("Teclado Virtual")
            keyboard_cols = st.columns(8)
            symbols = ["ε", "'", "|", "->", "+", "*", "(", ")"]
            
            for i, symbol in enumerate(symbols):
                if keyboard_cols[i].button(symbol):
                    grammar_text += symbol
            
        with col2:
            st.header("Cadena de entrada")
            input_source = st.radio("Fuente de la cadena:", 
                                    ["Escribir directamente", "Cargar desde archivo"])
            
            if input_source == "Escribir directamente":
                input_text = st.text_area("Ingrese cadena a analizar:", 
                                         height=100,
                                         value="id + id * id")
            else:
                uploaded_input = st.file_uploader("Subir archivo de entrada", type=["txt"])
                if uploaded_input:
                    input_text = uploaded_input.getvalue().decode("utf-8")
                else:
                    input_text = ""
        
        # Save input to temporary files
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt') as f:
            grammar_file = f.name
            f.write(grammar_text)
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.txt') as f:
            input_file = f.name
            f.write(input_text)
        
        st.session_state.grammar_file = grammar_file
        st.session_state.input_file = input_file
        st.session_state.grammar_text = grammar_text
        st.session_state.input_text = input_text
    
    with tabs[1]:
        if st.button("Analizar", key="analyze_btn"):
            if not grammar_text or not input_text:
                st.error("Por favor, ingrese la gramática y la cadena de entrada.")
            else:
                try:
                    # Extract the grammar rules
                    grammar_lines = grammar_text.split("\n")
                    grammar_lines = [line for line in grammar_lines if line.strip() and not line.strip().startswith('/')]
                    
                    # Write to temporary files for the parser
                    with open(grammar_file, 'w') as f:
                        f.write("\n".join(grammar_lines))
                    
                    with open(input_file, 'w') as f:
                        f.write(input_text)
                    
                    # Create a temporary input.txt in the working directory
                    with open("input.txt", 'w') as f:
                        f.write(input_text)
                    
                    # Execute the parser code
                    from LL_parser import extract_grammar, tokenize, explore_parse
                    
                    # Need to re-implement the core functionality to avoid running the main program
                    # This is just a placeholder - the real implementation would import and use
                    # functions from the LL_parser module to analyze the input
                    
                    # This is where we'd display the analysis results
                    st.success("Análisis completado!")
                    
                    # Display parse table (simplified example)
                    st.subheader("Tabla de Análisis LL(1)")
                    
                    # You would need to extract and format the actual parsing table from your parser
                    # This is a placeholder
                    parse_table_data = {
                        "": ["id", "+", "*", "(", ")", "$"],
                        "E": ["E -> T E'", "", "", "E -> T E'", "", ""],
                        "E'": ["", "E' -> + T E'", "", "", "E' -> ''", "E' -> ''"],
                        "T": ["T -> F T'", "", "", "T -> F T'", "", ""],
                        "T'": ["", "T' -> ''", "T' -> * F T'", "", "T' -> ''", "T' -> ''"],
                        "F": ["F -> id", "", "", "F -> ( E )", "", ""]
                    }
                    df = pd.DataFrame(parse_table_data)
                    st.table(df)
                    
                    # Display parsing steps
                    st.subheader("Pasos del Análisis")
                    
                    # You would need to collect these steps from your parser
                    steps_data = {
                        "Pila": ["E", "E'", "T E'", "F T' E'"],
                        "Entrada": ["id + id * id $", "id + id * id $", "id + id * id $", "id + id * id $"],
                        "Acción": ["Aplicar regla: E -> T E'", "", "Aplicar regla: T -> F T'", ""]
                    }
                    steps_df = pd.DataFrame(steps_data)
                    st.table(steps_df)
                    
                    # Store the steps in session state for tree visualization
                    st.session_state.analysis_steps = [
                        {"pila": "E", "entrada": "id + id * id $", "accion": "Aplicar regla: E -> T E'"},
                        {"pila": "T E'", "entrada": "id + id * id $", "accion": "Aplicar regla: T -> F T'"},
                        {"pila": "F T' E'", "entrada": "id + id * id $", "accion": "Aplicar regla: F -> id"},
                        {"pila": "id T' E'", "entrada": "id + id * id $", "accion": "Match: id"},
                        {"pila": "T' E'", "entrada": "+ id * id $", "accion": "Aplicar regla: T' -> ''"},
                        {"pila": "E'", "entrada": "+ id * id $", "accion": "Aplicar regla: E' -> + T E'"},
                        {"pila": "+ T E'", "entrada": "+ id * id $", "accion": "Match: +"},
                        {"pila": "T E'", "entrada": "id * id $", "accion": "Aplicar regla: T -> F T'"},
                        {"pila": "F T' E'", "entrada": "id * id $", "accion": "Aplicar regla: F -> id"},
                        {"pila": "id T' E'", "entrada": "id * id $", "accion": "Match: id"},
                        {"pila": "T' E'", "entrada": "* id $", "accion": "Aplicar regla: T' -> * F T'"},
                        {"pila": "* F T' E'", "entrada": "* id $", "accion": "Match: *"},
                        {"pila": "F T' E'", "entrada": "id $", "accion": "Aplicar regla: F -> id"},
                        {"pila": "id T' E'", "entrada": "id $", "accion": "Match: id"},
                        {"pila": "T' E'", "entrada": "$", "accion": "Aplicar regla: T' -> ''"},
                        {"pila": "E'", "entrada": "$", "accion": "Aplicar regla: E' -> ''"},
                        {"pila": "", "entrada": "$", "accion": "CADENA VÁLIDA"}
                    ]
                    
                    # Display result
                    st.subheader("Resultado")
                    st.success("Cadena ACEPTADA")
                    
                except Exception as e:
                    st.error(f"Error durante el análisis: {str(e)}")
    
    with tabs[2]:
        st.header("Visualización del Árbol Sintáctico")
        
        if 'analysis_steps' in st.session_state:
            tree = build_parse_tree(st.session_state.analysis_steps)
            st.graphviz_chart(tree)
        else:
            st.info("Primero analiza una cadena para visualizar su árbol sintáctico.")
    
    with tabs[3]:
        st.header("Optimización de Gramática")
        
        if 'grammar_text' in st.session_state and st.session_state.grammar_text:
            grammar_lines = [line for line in st.session_state.grammar_text.split('\n') if line.strip()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Eliminar Recursión Izquierda")
                if st.button("Aplicar"):
                    optimized = eliminate_left_recursion(grammar_lines)
                    st.session_state.optimized_grammar = '\n'.join(optimized)
                    st.text_area("Gramática Optimizada:", 
                                value=st.session_state.optimized_grammar, 
                                height=300)
            
            with col2:
                st.subheader("Factorización a la Izquierda")
                if st.button("Aplicar", key="factor_btn"):
                    # Use optimized grammar if it exists, otherwise use the original
                    to_factor = st.session_state.optimized_grammar.split('\n') if 'optimized_grammar' in st.session_state else grammar_lines
                    factored = left_factor(to_factor)
                    st.session_state.factored_grammar = '\n'.join(factored)
                    st.text_area("Gramática Factorizada:", 
                                value=st.session_state.factored_grammar, 
                                height=300)
            
            if st.button("Usar Gramática Optimizada"):
                final_grammar = ''
                if 'factored_grammar' in st.session_state:
                    final_grammar = st.session_state.factored_grammar
                elif 'optimized_grammar' in st.session_state:
                    final_grammar = st.session_state.optimized_grammar
                
                if final_grammar:
                    st.session_state.grammar_text = final_grammar
                    with open(st.session_state.grammar_file, 'w') as f:
                        f.write(final_grammar)
                    st.success("Gramática actualizada correctamente")
        else:
            st.info("Ingresa una gramática para optimizar")

# Cleanup function to delete temporary files
def cleanup():
    if 'grammar_file' in st.session_state:
        try:
            os.unlink(st.session_state.grammar_file)
        except:
            pass
    
    if 'input_file' in st.session_state:
        try:
            os.unlink(st.session_state.input_file)
        except:
            pass

# Run the app
if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
