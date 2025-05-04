import re
import json
import io
import graphviz
import copy

def extract_grammar(file_path):
    """Extract grammar rules from a file"""
    try:
        with open(file_path, "r", encoding="utf-8") as archivo:
            lines = archivo.readlines()
        return [line for line in lines if line.strip() and not line.strip().startswith('/')]
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as archivo:
            lines = archivo.readlines()
        return [line for line in lines if line.strip() and not line.strip().startswith('/')]

def tokenize(input_str, terminales):
    """Convert input string to tokens based on grammar terminals"""
    tokens = []
    i = 0
    while i < len(input_str):
        # Skip whitespace
        if input_str[i].isspace():
            i += 1
            continue
        
        # Try to match the longest terminal first
        matched = False
        
        # Sort terminals by length (longest first) to match the longest possible token
        sorted_terms = sorted(terminales, key=len, reverse=True)
        for term in sorted_terms:
            if input_str[i:i+len(term)] == term:
                tokens.append(term)
                i += len(term)
                matched = True
                break
        
        # If no terminal matches, consider it a single character token
        if not matched:
            tokens.append(input_str[i])
            i += 1
    
    return tokens

def explore_parse(input_str, grammar, tabla, start, terminales):
    """Explore the parsing of an input string"""
    # Tokenize the input string
    tokens = tokenize(input_str, terminales)
    tokens.append("$")  # Add end marker
    
    token_idx = 0
    pila = [start[0]]
    steps = []
    parse_tree = []  # For building the parse tree
    
    while True:
        pila_str = ' '.join(pila)
        entrada_str = ' '.join(tokens[token_idx:])
        
        if token_idx >= len(tokens) - 1 and len(pila) == 0:
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "CADENA VÁLIDA", "production": None})
            return True, steps
            
        if not pila:
            if token_idx < len(tokens) - 1:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "ERROR: Pila vacía pero entrada no completada", "production": None})
                return False, steps
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "CADENA VÁLIDA", "production": None})
                return True, steps
                
        if token_idx >= len(tokens):
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "ERROR: Entrada vacía pero pila no completada", "production": None})
            return False, steps
            
        current = pila[-1]
        current_token = tokens[token_idx]
        
        if grammar[current]["tipo"] in ["I", "V"]:
            if current_token in tabla[current] and tabla[current][current_token]:
                produccion = tabla[current][current_token][0]
                pila.pop()
                
                # Handle empty string case
                if produccion['Der'] != ['eps']:
                    pila.extend(reversed(produccion['Der']))
                
                # Create readable production string
                prod_str = produccion['Izq'] + ' -> ' + ' '.join(produccion['Der'])
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"Aplicar regla: {prod_str}", "production": produccion})
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: No hay producción para {current} con {current_token}", "production": None})
                return False, steps
        elif grammar[current]["tipo"] == "T":
            if current == current_token:
                pila.pop()
                token_idx += 1
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"Match: {current_token}", "production": None})
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: Se esperaba {current} pero se encontró {current_token}", "production": None})
                return False, steps
        else:
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: Símbolo {current} no reconocido", "production": None})
            return False, steps

def eliminate_left_recursion(grammar_text):
    """Eliminate left recursion from grammar"""
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
    
    # Apply the algorithm to eliminate direct left recursion
    new_rules = {}
    for A, productions in rules.items():
        alpha = []  # productions that start with A
        beta = []   # productions that don't start with A
        
        for prod in productions:
            prod_parts = prod.split()
            if prod_parts and prod_parts[0] == A:
                # This is a left recursive production
                alpha.append(' '.join(prod_parts[1:]) if len(prod_parts) > 1 else 'ε')
            else:
                beta.append(prod)
        
        if alpha:  # If there's left recursion
            A_prime = f"{A}'"
            new_rules[A] = []
            for b in beta:
                if b:
                    new_rules[A].append(f"{b} {A_prime}")
                else:
                    new_rules[A].append(A_prime)
            
            new_rules[A_prime] = []
            for a in alpha:
                if a != 'ε':
                    new_rules[A_prime].append(f"{a} {A_prime}")
            new_rules[A_prime].append('ε')
        else:
            new_rules[A] = productions
    
    # Convert back to grammar text
    result = []
    for non_terminal, productions in new_rules.items():
        if productions:
            result.append(f"{non_terminal} -> {' | '.join(productions)}")
    
    return '\n'.join(result)

def left_factorization(grammar_text):
    """Apply left factorization to grammar"""
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
    
    # Apply left factorization
    new_rules = {}
    for A, productions in rules.items():
        # Group productions by common prefix
        while True:
            # Find common prefixes
            prefixes = {}
            for prod in productions:
                prod_parts = prod.split()
                if not prod_parts:
                    if '' not in prefixes:
                        prefixes[''] = []
                    prefixes[''].append(prod)
                    continue
                    
                prefix = prod_parts[0]
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(prod)
            
            # Check if we need to factorize
            factorize = False
            for prefix, prods in prefixes.items():
                if len(prods) > 1:
                    factorize = True
                    break
            
            if not factorize:
                break
                
            # Apply factorization for the longest prefix
            new_productions = []
            for prefix, prods in list(prefixes.items()):
                if len(prods) > 1:
                    A_prime = f"{A}'"
                    new_suffix = []
                    
                    for prod in prods:
                        if prefix:
                            suffix = prod[len(prefix):].strip()
                            new_suffix.append(suffix if suffix else 'ε')
                        else:
                            new_suffix.append(prod)
                    
                    new_productions.append(f"{prefix} {A_prime}")
                    
                    if A_prime not in new_rules:
                        new_rules[A_prime] = []
                    new_rules[A_prime].extend(new_suffix)
                    
                    # Remove the factorized productions
                    productions = [p for p in productions if p not in prods]
                else:
                    new_productions.extend(prods)
            
            productions = new_productions
        
        new_rules[A] = productions
    
    # Convert back to grammar text
    result = []
    for non_terminal, productions in new_rules.items():
        if productions:
            result.append(f"{non_terminal} -> {' | '.join(productions)}")
    
    return '\n'.join(result)

def generate_parse_tree(parse_steps):
    """Generate a visual parse tree using graphviz"""
    # Create a new digraph
    dot = graphviz.Digraph('Parse Tree', format='png')
    
    # Configure for a clean, hierarchical tree layout with increased spacing
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('node', shape='circle', style='filled', color='black', fillcolor='lightblue', 
             fontname='Arial', height='0.6', width='0.6', fixedsize='true')
    dot.attr('edge', arrowhead='normal', color='black')
    # Increase spacing between nodes to prevent overlap
    dot.attr('graph', ranksep='0.8', nodesep='0.5', overlap='false', splines='true')
    
    # Add title
    dot.attr(label='Visualización del Árbol Sintáctico', labelloc='t', fontsize='20')
    
    # Track all actions: both rule applications and token matches
    all_actions = []
    for step in parse_steps:
        action = step.get('accion', '')
        if action.startswith('Aplicar regla:'):
            rule = action.replace('Aplicar regla: ', '')
            left_side, right_side = rule.split('->')
            all_actions.append(('rule', left_side.strip(), right_side.strip()))
        elif action.startswith('Match:'):
            token = action.replace('Match: ', '')
            all_actions.append(('match', token))
    
    # Initialize tracking variables
    node_counter = 0
    node_map = {}  # Maps symbols to their node IDs
    parent_stack = []
    terminal_nodes = []  # To track terminal nodes in order they are matched
    
    # Find the start symbol (from the first rule application)
    start_symbol = None
    for action_type, *args in all_actions:
        if action_type == 'rule':
            start_symbol = args[0]
            break
    
    if not start_symbol:
        return dot  # Empty tree if no rules found
    
    # Create root node
    root_id = f"node_{node_counter}"
    node_counter += 1
    dot.node(root_id, start_symbol)
    parent_stack.append((root_id, start_symbol))
    
    # Process all actions in order
    for action_type, *args in all_actions:
        if action_type == 'rule':
            left_side, right_side = args
            
            # Find and remove the non-terminal being expanded from the stack
            parent_id = None
            parent_idx = -1
            for idx, (node_id, symbol) in enumerate(parent_stack):
                if symbol == left_side:
                    parent_id = node_id
                    parent_idx = idx
                    break
            
            if parent_id is None:
                continue  # Skip if parent not found
            
            # Remove the expanded non-terminal
            parent_stack.pop(parent_idx)
            
            # Process the right side of the production
            symbols = right_side.split()
            if right_side == 'ε':  # Special case for epsilon
                symbols = ['ε']
            
            # Add child nodes in reverse order to stack (for left-to-right processing)
            children_ids = []
            for symbol in symbols:
                child_id = f"node_{node_counter}"
                node_counter += 1
                
                # Add node and edge
                dot.node(child_id, symbol)
                dot.edge(parent_id, child_id)
                children_ids.append(child_id)
                
                # Add to stack in reverse order for left-to-right expansions
                if symbol[0].isupper() or "'" in symbol:  # Non-terminal
                    parent_stack.append((child_id, symbol))
                elif symbol != 'ε':  # Terminal but not epsilon
                    terminal_nodes.append((child_id, symbol))
    
    # Use subgraphs to ensure terminals are aligned at the bottom
    with dot.subgraph() as s:
        s.attr(rank='same')
        for node_id, _ in terminal_nodes:
            s.node(node_id)
    
    return dot

# Main program
def parse_grammar_and_analyze(grammar_file="grammar.txt", input_file="input.txt", return_steps=False):
    try:
        grammar_lines = extract_grammar(grammar_file)
        
        # Reemplazar épsilon unicode con texto plano "eps" para evitar errores de codificación
        epsilon = 'eps'
        
        # Parse the grammar
        start = [grammar_lines[0].split('->')[0].strip()]
        
        reglas = {}
        variables = []
        terminales = []
        
        # First, identify variables and terminals
        for line in grammar_lines:
            line = line.strip()
            if not line or '->' not in line:
                continue
                
            parts = line.split('->')
            var = parts[0].strip()
            
            # Don't add the start symbol to variables list to avoid duplication
            if var not in variables and var not in start:
                variables.append(var)
            # Extract alternatives
            right_side = parts[1].strip()
            alternatives = [alt.strip() for alt in right_side.split('|')]
            
            for alt_idx, alt in enumerate(alternatives):
                regla_id = f'rule_{var}_{alt_idx}'
                
                # Handle empty string case
                if alt == "''" or alt == '""' or alt == '' or alt == 'ε':
                    reglas[regla_id] = {'Izq': var, 'Der': [epsilon]}
                    continue
                
                # Process multi-character symbols
                i = 0
                symbols = []
                while i < len(alt):
                    if alt[i].isspace():
                        i += 1
                        continue
                        
                    # Handle variables with apostrophe (like E')
                    if i+1 < len(alt) and alt[i+1] == "'":
                        symbol = alt[i:i+2]
                        if symbol not in variables:
                            variables.append(symbol)
                        symbols.append(symbol)
                        i += 2
                        continue
                    
                    # General case: handle any multi-character terminal (sequence of lowercase letters)
                    if alt[i].islower():
                        start_pos = i
                        while i < len(alt) and alt[i].islower():
                            i += 1
                        symbol = alt[start_pos:i]
                        if symbol not in terminales:
                            terminales.append(symbol)
                        symbols.append(symbol)
                        continue
                    
                    # Handle special characters
                    if alt[i] in '+-*/()':
                        if alt[i] not in terminales:
                            terminales.append(alt[i])
                        symbols.append(alt[i])
                        i += 1
                        continue
                        
                    # Handle other symbols
                    symbol = alt[i]
                    if symbol.isupper():
                        if symbol not in variables:
                            variables.append(symbol)
                    else:
                        if symbol not in terminales and symbol != epsilon:
                            terminales.append(symbol)
                    symbols.append(symbol)
                    i += 1
                    
                reglas[regla_id] = {'Izq': var, 'Der': symbols}
        
        # Initialize grammar structure
        grammar = {}
        
        # First add start symbols to ensure they're in the grammar
        for s in start:
            grammar[s] = {"tipo": "I", "first": [], "follow": ["$"], "nullable": False}
            
        # Then add other variables
        for var in variables:
            if var not in grammar:  # Skip if already added (like start symbol)
                grammar[var] = {"tipo": "V", "first": [], "follow": [], "nullable": False}
            
        for term in terminales:
            grammar[term] = {"tipo": "T", "first": [term], "nullable": False}
        
        if epsilon not in grammar:
            grammar[epsilon] = {"tipo": "E", "first": [], "nullable": True}
        
        # Calculate nullable symbols
        changed = True
        while changed:
            changed = False
            for r in reglas.values():
                var = r['Izq']
                if not grammar[var]["nullable"]:
                    # Check if all symbols in the right side are nullable
                    all_nullable = all(grammar[s]["nullable"] for s in r['Der']) if r['Der'] else True
                    if all_nullable or epsilon in r['Der']:
                        grammar[var]["nullable"] = True
                        changed = True
        
        # Calculate FIRST sets
        changed = True
        while changed:
            changed = False
            for r in reglas.values():
                var = r['Izq']
                initial_size = len(grammar[var]["first"])
                
                # For empty productions
                if epsilon in r['Der']:
                    if epsilon not in grammar[var]["first"]:
                        grammar[var]["first"].append(epsilon)
                        changed = True
                    continue
                    
                # For non-empty productions
                for i, symbol in enumerate(r['Der']):
                    # Add FIRST of current symbol
                    for f in grammar[symbol]["first"]:
                        if f != epsilon and f not in grammar[var]["first"]:
                            grammar[var]["first"].append(f)
                            changed = True
                    
                    # If not nullable, stop
                    if not grammar[symbol]["nullable"]:
                        break
                        
                # If all symbols are nullable, add epsilon
                if all(grammar[s]["nullable"] for s in r['Der']) and r['Der']:
                    if epsilon not in grammar[var]["first"]:
                        grammar[var]["first"].append(epsilon)
                        changed = True
                        
                if len(grammar[var]["first"]) > initial_size:
                    changed = True
        
        # Calculate FOLLOW sets
        changed = True
        while changed:
            changed = False
            for r in reglas.values():
                var = r['Izq']
                der = r['Der']
                
                for i, symbol in enumerate(der):
                    if grammar[symbol]["tipo"] in ["V", "I"]:
                        initial_size = len(grammar[symbol]["follow"])
                        
                        # Case 1: A -> αBβ, add FIRST(β) - {ε} to FOLLOW(B)
                        if i < len(der) - 1:
                            beta = der[i+1:]
                            first_beta = []
                            
                            # Calculate FIRST of beta
                            all_nullable = True
                            for s in beta:
                                for f in grammar[s]["first"]:
                                    if f != epsilon and f not in first_beta:
                                        first_beta.append(f)
                                if not grammar[s]["nullable"]:
                                    all_nullable = False
                                    break
                            
                            # Add FIRST(β) - {ε} to FOLLOW(B)
                            for f in first_beta:
                                if f not in grammar[symbol]["follow"]:
                                    grammar[symbol]["follow"].append(f)
                                    changed = True
                            
                            # Case 2: A -> αBβ where β can derive ε, add FOLLOW(A) to FOLLOW(B)
                            if all_nullable:
                                for f in grammar[var]["follow"]:
                                    if f not in grammar[symbol]["follow"]:
                                        grammar[symbol]["follow"].append(f)
                                        changed = True
                        
                        # Case 3: A -> αB, add FOLLOW(A) to FOLLOW(B)
                        elif i == len(der) - 1:
                            for f in grammar[var]["follow"]:
                                if f not in grammar[symbol]["follow"]:
                                    grammar[symbol]["follow"].append(f)
                                    changed = True
                        
                        if len(grammar[symbol]["follow"]) > initial_size:
                            changed = True
        
        # Build LL parsing table
        tabla = {}
        for i in start + variables:
            tabla[i] = {}
            for j in terminales + ["$"]:
                tabla[i][j] = []
        
        for r_id, r in reglas.items():
            var = r['Izq']
            der = r['Der']
            
            # Handle epsilon productions
            if epsilon in der:
                for follow_symbol in grammar[var]["follow"]:
                    if r not in tabla[var][follow_symbol]:
                        tabla[var][follow_symbol].append(r)
            else:
                # Calculate FIRST of the right side
                first_der = []
                all_nullable = True
                
                for symbol in der:
                    for f in grammar[symbol]["first"]:
                        if f != epsilon and f not in first_der:
                            first_der.append(f)
                    if not grammar[symbol]["nullable"]:
                        all_nullable = False
                        break
                
                # Add entry to table for each terminal in FIRST
                for first_symbol in first_der:
                    if r not in tabla[var][first_symbol]:
                        tabla[var][first_symbol].append(r)
                
                # If right side can derive epsilon, add entry for each terminal in FOLLOW
                if all_nullable:
                    for follow_symbol in grammar[var]["follow"]:
                        if r not in tabla[var][follow_symbol]:
                            tabla[var][follow_symbol].append(r)
        
        # Preparar salida para análisis
        output = io.StringIO()
        
        # Display grammar information and parsing table
        output.write("TABLA DE SÍMBOLOS\n\n")
        output.write(f"{'Símbolo':<10} {'Tipo':<8} {'NULLABLE':<10} {'FIRST':<20} {'FOLLOW':<20}\n")
        output.write("-" * 70 + "\n")
        
        for simbolo, datos in grammar.items():
            if simbolo == epsilon:  # Skip epsilon in symbol table display
                continue
            tipo = datos['tipo']
            nullable = "Sí" if datos.get('nullable', False) else "No"
            
            # Show epsilon as 'ε' in FIRST sets for display
            first_set = datos.get('first', [])
            first = ", ".join(['ε' if f == epsilon else f for f in first_set])
            
            follow = ", ".join(datos.get('follow', [])) if 'follow' in datos else "-"
            output.write(f"{simbolo:<10} {tipo:<8} {nullable:<10} {first:<20} {follow:<20}\n")
        
        output.write("\n\nTABLA DE ANÁLISIS LL(1)\n")
        output.write(f"{'Non-Terminal':20}")
        
        # Display only actual terminals (not epsilon) in the table header
        display_terminals = [t for t in terminales + ["$"] if t != epsilon]
        for t in display_terminals:
            output.write(f"{t:<20}")
        output.write("\n")
        
        output.write("-" * (20 + 20 * len(display_terminals)) + "\n")
        
        # Use set to avoid duplicates when displaying
        displayed_symbols = []
        for nt in start + variables:
            if nt in displayed_symbols:
                continue
            displayed_symbols.append(nt)
            
            output.write(f"{nt:20}")
            for t in display_terminals:  # Only iterate over actual terminals
                producciones = tabla[nt][t]
                if producciones:
                    # Create a cleaner, single production string for each cell
                    # Only show the first production if there are multiple
                    if producciones:
                        p = producciones[0]
                        production_str = f"{p['Izq']} -> {' '.join(p['Der']).replace('eps', 'ε')}"
                    else:
                        production_str = "-"
                    output.write(f"{production_str:<20}")
                else:
                    output.write(f"{'-':<20}")
            output.write("\n")
        
        # Parse input string
        try:
            with open(input_file, "r", encoding="utf-8") as code:
                input_text = code.readlines()[0].strip()
        except UnicodeDecodeError:
            with open(input_file, "r", encoding="latin-1") as code:
                input_text = code.readlines()[0].strip()
        
        output.write("\n\nANALISIS DE LA CADENA\n")
        success, steps = explore_parse(input_text, grammar, tabla, start, terminales)
        
        # Print the parsing steps in tabular format
        output.write(f"{'PILA':<30} {'ENTRADA':<30} {'ACCIÓN':<50}\n")
        output.write("-" * 110 + "\n")
        
        # Print the parsing steps
        for step in steps:
            output.write(f"{step['pila']:<30} {step['entrada']:<30} {step['accion']:<50}\n")
        
        output.write("\nResultado del análisis: " + ("ACEPTADA" if success else "RECHAZADA") + "\n")
        
        if return_steps:
            return output.getvalue(), steps
        return output.getvalue()
    except Exception as e:
        return f"Error en el análisis: {str(e)}\n{type(e).__name__}: {e}"

if __name__ == "__main__":
    result = parse_grammar_and_analyze()
    print(result)