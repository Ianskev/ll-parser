import graphviz

# Create a class diagram
diagram = graphviz.Digraph('LL Parser System', filename='class_diagram.gv', format='png')
diagram.attr('node', shape='box')

# Define the classes
diagram.node('LL_Parser', '''LL_Parser
--
+ extract_grammar(file_path)
+ tokenize(input_str, terminales)
+ explore_parse(input_str, grammar, tabla, start, terminales)
''')

diagram.node('GrammarOptimizer', '''GrammarOptimizer
--
+ eliminate_left_recursion(rules)
+ left_factor(rules)
''')

diagram.node('ParseTreeBuilder', '''ParseTreeBuilder
--
+ build_parse_tree(steps)
''')

diagram.node('StreamlitUI', '''StreamlitUI
--
+ display_grammar_input()
+ display_parse_table()
+ display_parse_tree()
+ display_optimized_grammar()
''')

# Define the relationships
diagram.edge('StreamlitUI', 'LL_Parser', label='uses')
diagram.edge('StreamlitUI', 'GrammarOptimizer', label='uses')
diagram.edge('StreamlitUI', 'ParseTreeBuilder', label='uses')
diagram.edge('LL_Parser', 'ParseTreeBuilder', label='provides data to')

# Render the diagram
diagram.render(directory='diagrams', view=True)

print("Class diagram generated.")
