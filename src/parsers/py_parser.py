import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import json
import re
import math

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def calculate_cyclomatic_complexity(code):
    """
    Calculates McCabe's Cyclomatic Complexity using Tree-Sitter for Python.
    Formula: M = E - N + 2P
    Simplified for single function: 1 + number of decision points.
    Decision points: if, while, for, except, with (with branching), 
    logical 'and', 'or', and list comprehensions.
    """
    if not code or not isinstance(code, str):
        return 1
        
    tree = parser.parse(bytes(code, "utf8"))
    
    # Decision points in Python tree-sitter grammar
    decision_nodes = {
        'if_statement',
        'while_statement',
        'for_statement',
        'conditional_expression', # ternary
        'boolean_operator', # and/or
        'except_clause',
        'list_comprehension',
        'dictionary_comprehension',
        'set_comprehension',
        'generator_expression'
    }
    
    complexity = 1
    
    def traverse(node):
        nonlocal complexity
        if node.type in decision_nodes:
            complexity += 1
        
        for child in node.children:
            traverse(child)
            
    traverse(tree.root_node)
    return complexity


def calculate_nesting_depth(code):
    """
    Calculates the maximum nesting depth of control structures using Tree-Sitter.
    This captures architectural complexity beyond simple branch counting.
    """
    if not code or not isinstance(code, str):
        return 0

    tree = parser.parse(bytes(code, "utf8"))
    
    nesting_nodes = {
        'if_statement', 'while_statement', 'for_statement',
        'try_statement', 'with_statement',
        'function_definition', 'class_definition'
    }
    
    max_depth = 0
    
    def traverse(node, depth):
        nonlocal max_depth
        current_depth = depth
        if node.type in nesting_nodes:
            current_depth = depth + 1
            max_depth = max(max_depth, current_depth)
        
        for child in node.children:
            traverse(child, current_depth)
    
    traverse(tree.root_node, 0)
    return max_depth


def extract_ast_features(code):
    """
    Extracts AST-based structural features from Python code.
    Returns a dict of features for use as instruments or controls.
    """
    if not code or not isinstance(code, str):
        return {
            'ast_cyclomatic': 1,
            'ast_nesting_depth': 0,
            'ast_function_count': 0,
            'ast_class_count': 0,
        }
    
    tree = parser.parse(bytes(code, "utf8"))
    
    func_count = 0
    class_count = 0
    
    def count_nodes(node):
        nonlocal func_count, class_count
        if node.type == 'function_definition':
            func_count += 1
        elif node.type == 'class_definition':
            class_count += 1
        for child in node.children:
            count_nodes(child)
    
    count_nodes(tree.root_node)
    
    return {
        'ast_cyclomatic': calculate_cyclomatic_complexity(code),
        'ast_nesting_depth': calculate_nesting_depth(code),
        'ast_function_count': func_count,
        'ast_class_count': class_count,
    }


def extract_unbiased_features(instruction):
    """
    Extracts deep structural intent from the instruction.
    Bypasses simple keyword counts to find nested requirements.
    """
    if not instruction:
        return {
            'nesting_intent': 0,
            'algo_intent': 0,
            'instruction_complexity_score': 0.0,
        }
        
    # Find nested logic requirements (e.g., "if X then if Y")
    nested_patterns = [
        r'nested', r'recursive', r'recursion', r'inner', r'inside',
        r'sub-task', r'hierarchy', r'depth', r'tree'
    ]
    
    # Look for specific algorithmic complexity indicators
    algo_patterns = [
        r'sort', r'search', r'filter', r'map', r'reduce', r'aggregate',
        r'validate', r'parse', r'convert', r'transform'
    ]
    
    tokens = instruction.lower().split()
    
    return {
        'nesting_intent': sum(1 for p in nested_patterns if re.search(p, instruction, re.I)),
        'algo_intent': sum(1 for p in algo_patterns if re.search(p, instruction, re.I)),
        'instruction_complexity_score': len(set(tokens)) / (math.log(len(tokens) + 1) if len(tokens) > 0 else 1)
    }


if __name__ == "__main__":
    test_code = """
def complex_func(x):
    if x > 0:
        for i in range(x):
            print(i)
    elif x == 0:
        return 0
    else:
        return -1
    """
    print(f"Test Code Complexity: {calculate_cyclomatic_complexity(test_code)}")
    print(f"Test Nesting Depth: {calculate_nesting_depth(test_code)}")
    print(f"Test AST Features: {extract_ast_features(test_code)}")
