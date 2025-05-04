from LL_parser import parse_grammar_and_analyze
import tempfile
import os

def test_struct_grammar():
    # Create temporary files for grammar and input
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        grammar = """Struct  -> struct Nombre { Comps }
Nombre  -> id
Comps   -> Comp Comps'
Comps'  -> ; Comp Comps' | ε
Comp    -> Type id
Type    -> Typep | struct id | Pointer
Typep   -> int | char | bool | float
Pointer -> * id"""
        f.write(grammar)
        grammar_file = f.name
        
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write("struct id { int id ; struct id id ; * id id }")
        input_file = f.name
    
    # Test if the grammar parses correctly
    try:
        print("Testing struct grammar...")
        result = parse_grammar_and_analyze(grammar_file, input_file)
        print(result)
        print("Testing completed.")
        if "ACEPTADA" in result:
            print("SUCCESS: The grammar was accepted correctly.")
        else:
            print("FAILED: The grammar was rejected.")
    finally:
        # Clean up temporary files
        os.unlink(grammar_file)
        os.unlink(input_file)

if __name__ == "__main__":
    test_struct_grammar()
