import streamlit as st
import tempfile
import os
import base64
import shutil
import matplotlib.pyplot as plt
import networkx as nx
from LL_parser import parse_grammar_and_analyze, eliminate_left_recursion, left_factorization, generate_parse_tree

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
                    
                    with st.container():
                        st.text(result)
                    
                    # Botón para descargar resultados
                    st.download_button(
                        label="Descargar Resultados",
                        data=result,
                        file_name="analisis_ll1.txt",
                        mime="text/plain",
                    )
                    
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

def create_fallback_tree_visualization(parse_steps):
    """Create a simple parse tree visualization using matplotlib when Graphviz is not available"""
    productions = [step for step in parse_steps if step.get('production')]
    
    if not productions:
        st.warning("No hay suficientes datos para generar el árbol de derivación.")
        return
    
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Track nodes and relationships
        nodes = {}
        node_counter = 0
        root_node = None
        
        # First pass: identify all nodes
        for step in productions:
            prod = step.get('production')
            if prod:
                left_side = prod['Izq']
                right_side = prod['Der']
                
                # Add parent node if not exists
                if left_side not in nodes:
                    node_id = f"{left_side}_{node_counter}"
                    nodes[left_side] = node_id
                    G.add_node(node_id, label=left_side)
                    if root_node is None:  # First node becomes root
                        root_node = node_id
                    node_counter += 1
                
                # Add child nodes
                for symbol in right_side:
                    symbol_key = f"{symbol}_{node_counter}"
                    if symbol == 'eps':
                        symbol_display = 'ε'
                    else:
                        symbol_display = symbol
                    
                    G.add_node(symbol_key, label=symbol_display)
                    G.add_edge(nodes[left_side], symbol_key)
                    nodes[symbol + str(node_counter)] = symbol_key
                    node_counter += 1
        
        # Draw the graph with a pure networkx layout
        plt.figure(figsize=(12, 8))
        
        try:
            # Try hierarchical layout first (best for trees)
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        except (ImportError, Exception):
            try:
                # Try Kamada-Kawai layout as first fallback (often good for trees)
                pos = nx.kamada_kawai_layout(G)
            except Exception:
                # Use spring layout as final fallback (works in all NetworkX installations)
                pos = nx.spring_layout(G, k=0.8, iterations=100)
        
        # Draw nodes with labels
        nx.draw(G, pos, with_labels=False, node_color='lightblue', 
                node_size=700, arrows=True, edge_color='black', arrowsize=20)
        
        # Add node labels
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title("Árbol de Derivación")
        plt.axis('off')
        
        # Display in Streamlit
        st.pyplot(plt)
        
        st.caption("Nota: Esta es una visualización simplificada. Para mejor calidad, instala Graphviz.")
        
        # Provide installation instructions for all required packages
        with st.expander("Solucionar problemas de visualización"):
            st.markdown("""
            Para una óptima visualización del árbol de derivación, instale los siguientes paquetes:

            ```bash
            pip install graphviz pygraphviz matplotlib networkx
            ```
            
            En algunos sistemas también necesitará instalar Graphviz como programa:
            
            - **Windows**: 
              - Descarga desde [graphviz.org](https://graphviz.org/download/) 
              - O usa: `winget install graphviz`
            
            - **macOS**: 
              - `brew install graphviz`
            
            - **Linux**: 
              - `sudo apt-get install graphviz libgraphviz-dev pkg-config`
            
            Después de la instalación, reinicie la aplicación.
            """)
    except Exception as e:
        st.error(f"Error al generar la visualización alternativa: {str(e)}")
        
        # Mostrar instrucciones para solucionar el problema
        st.markdown("""
        ### Visualización básica de la derivación
        
        No se pudo generar una visualización gráfica del árbol debido a limitaciones de dependencias.
        A continuación se muestra una representación textual de las derivaciones:
        """)
        
        # Create a simple text-based representation
        prod_table = []
        for step in productions:
            prod = step.get('production')
            if prod:
                left_side = prod['Izq']
                right_side = ' '.join(prod['Der']).replace('eps', 'ε')
                prod_table.append({"No terminal": left_side, "Derivación": f"{left_side} → {right_side}"})
        
        # Display as a table
        if prod_table:
            st.table(prod_table)
        else:
            st.write("No hay información de derivación disponible.")

if __name__ == "__main__":
    main()
