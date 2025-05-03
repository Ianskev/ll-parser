import re
import json

def extract_grammar(file_path):
    """Extract grammar rules from a file"""
    archivo = open(file_path, "r")
    lines = archivo.readlines()
    archivo.close()
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
    
    while True:
        pila_str = ' '.join(pila)
        entrada_str = ' '.join(tokens[token_idx:])
        
        if token_idx >= len(tokens) - 1 and len(pila) == 0:
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "CADENA VÁLIDA"})
            return True, steps
            
        if not pila:
            if token_idx < len(tokens) - 1:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "ERROR: Pila vacía pero entrada no completada"})
                return False, steps
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "CADENA VÁLIDA"})
                return True, steps
                
        if token_idx >= len(tokens):
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": "ERROR: Entrada vacía pero pila no completada"})
            return False, steps
            
        current = pila[-1]
        current_token = tokens[token_idx]
        
        if grammar[current]["tipo"] in ["I", "V"]:
            if current_token in tabla[current] and tabla[current][current_token]:
                produccion = tabla[current][current_token][0]
                pila.pop()
                
                # Handle empty string case
                if produccion['Der'] != ['ε']:
                    pila.extend(reversed(produccion['Der']))
                
                # Create readable production string
                prod_str = produccion['Izq'] + ' -> ' + ' '.join(produccion['Der'])
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"Aplicar regla: {prod_str}"})
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: No hay producción para {current} con {current_token}"})
                return False, steps
        elif grammar[current]["tipo"] == "T":
            if current == current_token:
                pila.pop()
                token_idx += 1
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"Match: {current_token}"})
            else:
                steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: Se esperaba {current} pero se encontró {current_token}"})
                return False, steps
        else:
            steps.append({"pila": pila_str, "entrada": entrada_str, "accion": f"ERROR: Símbolo {current} no reconocido"})
            return False, steps

# Main program
grammar_lines = extract_grammar("grammar.txt")
epsilon = 'ε'  # We'll still use this internally, but recognize '' in the grammar

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
        if alt == "''" or alt == '""' or alt == '':
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

# Display grammar information and parsing table
print("TABLA DE SÍMBOLOS")
print("\n")
print(f"{'Símbolo':<10} {'Tipo':<8} {'NULLABLE':<10} {'FIRST':<20} {'FOLLOW':<20}")
print("-" * 70)

for simbolo, datos in grammar.items():
    if simbolo == epsilon:
        continue
    tipo = datos['tipo']
    nullable = "Sí" if datos.get('nullable', False) else "No"
    first = ", ".join(datos.get('first', []))
    follow = ", ".join(datos.get('follow', [])) if 'follow' in datos else "-"
    print(f"{simbolo:<10} {tipo:<8} {nullable:<10} {first:<20} {follow:<20}")

print("\n\nTABLA DE ANÁLISIS LL(1)")
print(f"{'':20}", end="")
for t in terminales + ["$"]:
    print(f"{t:<20}", end="")
print()

print("-" * (20 + 20 * len(terminales + ["$"])))

# Use set to avoid duplicates when displaying
displayed_symbols = []
for nt in start + variables:
    if nt in displayed_symbols:
        continue
    displayed_symbols.append(nt)
    
    print(f"{nt:20}", end="")
    for t in terminales + ["$"]:
        producciones = tabla[nt][t]
        if producciones:
            produccion_strs = [f"{p['Izq']} -> {' '.join(p['Der'])}" for p in producciones]
            print(f"{' / '.join(produccion_strs):<20}", end="")
        else:
            print(f"{'-':<20}", end="")
    print()

# Parse input string
code = open("input.txt", "r")
input_text = code.readlines()[0].strip()
code.close()

print("\n\nANALISIS DE LA CADENA")
success, steps = explore_parse(input_text, grammar, tabla, start, terminales)

# Print the parsing steps
for step in steps:
    print(f"{step['pila']:<30} {step['entrada']:<30} {step['accion']}")

print("\nResultado del análisis:", "ACEPTADA" if success else "RECHAZADA")