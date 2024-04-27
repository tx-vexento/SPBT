from ist_utils import text, print_children
from collections import defaultdict
from transform.lang import get_lang

declaration_map = {'c': 'declaration', 'java': 'local_variable_declaration', 'c_sharp': 'local_declaration_statement'}
block_map = {'c': 'compound_statement', 'java': 'block', 'c_sharp': 'block'}

def get_for_info(node):
    # 提取for循环的abc信息，for(a;b;c)以及后面接的语句
    i, abc = 0, [None, None, None, None]
    for child in node.children:
        if child.type in [';', ')', declaration_map[get_lang()]]:
            if child.type == declaration_map[get_lang()]:
                abc[i] = child
            if child.prev_sibling is None: return
            if child.prev_sibling.type not in ['(', ';']:
                abc[i] = child.prev_sibling
            i += 1
        if child.prev_sibling and child.prev_sibling.type == ')' and i == 3:
            abc[3] = child
    return abc

def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    if len(code) <= i: return indent
    while i >= 0 and code[i] != '\n':
        if code[i] == ' ':
            indent += 1
        elif code[i] == '\t':
            indent += 4
        i -= 1
    return indent

def contain_id(node, contain):
    # 返回node节点子树中的所有变量名
    if node.child_by_field_name('index'):   # a[i] < 2中的index：i
        contain.add(text(node.child_by_field_name('index')))
    if node.type == 'identifier' and node.parent.type not in ['subscript_expression', 'call_expression']:   # a < 2中的a
        contain.add(text(node))
    if not node.children:
        return
    for n in node.children:
        contain_id(n, contain)

'''==========================匹配========================'''
def match_for(root):
    def check(node):
        if node.type == 'for_statement':
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_while(root):
    def check(node):
        if node.type == 'while_statement':
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_do_while(root):
    def check(node):
        if node.type == 'do_statement' and 'while' in text(node):
            return True
        return False
    res = []
    def match(u):
        if check(u): res.append(u)
        for v in u.children:
            match(v)
    match(root)
    return res

def match_loop(root):
    return match_for(root) + match_while(root) + match_do_while(root)

'''==========================替换========================'''
def convert_for(node, code):
    if node.type == 'while_statement':
        # for(a;b;c)
        a = b = c = None
        if get_lang() == 'c_sharp':
            b = node.children[4].children[1]
        else:
            b = node.children[1].children[1]
        id, prev_id, clause_id = set(), set(), set()
        contain_id(b, id)
        if len(id) == 0:
            return
        id = list(id)[0]
        prev = node.prev_sibling
        if prev.type == declaration_map[get_lang()] and prev.child_count == 3 or prev.type == 'expression_statement' and prev.children[0].type in ['update_expression', 'assignment_expression']:
            contain_id(node.prev_sibling, prev_id)
        if len(prev_id):
            for u in list(prev_id):
                if u in id:    # 如果前面一句是声明或者赋值并且和循环的id一样
                    a = node.prev_sibling
        for u in node.children[2].children[1: -1]:
            if u.type == 'expression_statement' and u.parent.type not in ['if_statement', 'for_statement', 'else_clause', 'while_statement']:  # 是i++,i+=1这种表达式，且不能在从句里面
                if u.children[0].type in ['update_expression', 'assignment_expression']:
                    contain_id(u.children[0], clause_id)
                    if len(clause_id) == 1 and id in clause_id: # 从句里面如果有++或者赋值运算并且变量名为id，则添加c
                        c = u
                        break
        res = [(node.children[1].end_byte, node.children[0].start_byte)]
        if a:
            res.append((a.end_byte, a.prev_sibling.end_byte))
        if c:
            res.append((c.end_byte, c.prev_sibling.end_byte))
        text_a = text(a) if a else ''
        for_str = f"for({text_a}{'; ' if ';' not in text_a else ' '}{text(b)}; {text(c).replace(';', '') if c else ''})"
        res.append((node.start_byte, for_str))
        return res
    elif node.type == 'do_statement':
        # for(a;b;c)
        a = b = c = None
        for v in node.children:
            if v.type == 'parenthesized_expression':
                b = v.children[1]
                break
        id, prev_id, clause_id = set(), set(), set()
        contain_id(b, id)
        if len(id) == 0:
            return
        id = list(id)[0]
        prev = node.prev_sibling
        if prev.type == declaration_map[get_lang()] and prev.child_count == 3 or prev.type == 'expression_statement' and prev.children[0].type in ['update_expression', 'assignment_expression']:
            contain_id(node.prev_sibling, prev_id)
        if len(prev_id):
            for u in list(prev_id):
                if u in id:    # 如果前面一句是声明或者赋值并且和循环的id一样
                    a = node.prev_sibling
        for u in node.children[1].children[1: -1]:
            if u.type == 'expression_statement' and u.parent.type not in ['if_statement', 'for_statement', 'else_clause', 'while_statement']:  # 是i++,i+=1这种表达式，且不能在从句里面
                if u.children[0].type in ['update_expression', 'assignment_expression']:
                    contain_id(u.children[0], clause_id)
                    if len(clause_id) == 1 and id in clause_id: # 从句里面如果有++或者赋值运算并且变量名为id，则添加c
                        c = u
                        break
        res = [(node.children[0].end_byte, node.children[0].start_byte),
               (node.children[4].end_byte, node.children[2].start_byte)]
        if a:
            res.append((a.end_byte, a.prev_sibling.end_byte))
        if c:
            res.append((c.end_byte, c.prev_sibling.end_byte))
        text_a = text(a) if a else ''
        for_str = f"for({text_a}{'; ' if ';' not in text_a else ' '}{text(b)}; {text(c).replace(';', '') if c else ''})"
        res.append((node.start_byte, for_str))

        return res

def count_for(root):
    nodes= match_for(root)
    return len(nodes)

def convert_while(node, code):
    node_info = defaultdict(type(node))
    # a while(b) c
    if node.type == 'for_statement':
        res, add_bracket = [], None
        abc = get_for_info(node)
        for c in node.children:
            if c.type == 'block':
                node_info['block'] = c
                break
        res = [(node_info['block'].start_byte - 1, node.start_byte)]    # 删除for(a;b;c)
        if abc[0] is not None:  # 如果有a
            indent = get_indent(node.start_byte, code)
            if abc[0].type != declaration_map[get_lang()]:
                res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
            else:
                res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
        if abc[2] is not None and abc[3] is not None:  # 如果有c
            if abc[3].type == block_map[get_lang()]:     # 复合语句在第一句插入if b break
                last_expression_node = abc[3].children[-2]
            else:       # 如果是单行，后面加上了花括号，在就在expression的开始位置插入
                last_expression_node = abc[3]
            indent = get_indent(last_expression_node.start_byte, code)
            res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
        while_str = f"while({text(abc[1]) if abc[1] else ''})"
        res.append((node.children[0].start_byte, while_str))
        if add_bracket:
            res.extend(add_bracket)
        return res
    elif node.type == 'do_statement':
        condition_node = node.children[3]
        return [(node.children[0].end_byte, node.children[0].start_byte),
                (node.children[4].end_byte, node.children[2].start_byte),
                (node.children[0].start_byte, f"whlie{text(condition_node)}")]

def count_while(root):
    nodes= match_while(root)
    return len(nodes)

def convert_do_while(node, code):
    node_info = defaultdict(type(node))
    if node.type == 'for_statement':
        # a while(b) c
        res, add_bracket = [], None
        abc = get_for_info(node)
        if abc is None: return
        for c in node.children:
            if c.type == 'block':
                node_info['block'] = c
                break
        res = [(node_info['block'].start_byte - 1, node.start_byte)]    # 删除for(a;b;c)
        if abc[0] is not None:  # 如果有a
            indent = get_indent(node.start_byte, code)
            if abc[0].type != declaration_map[get_lang()]:
                res.append((node.start_byte, text(abc[0]) + f';\n{indent * " "}'))
            else:
                res.append((node.start_byte, text(abc[0]) + f'\n{indent * " "}'))
        if abc[2] is not None and abc[3] is not None:  # 如果有c
            if abc[3].type == block_map[get_lang()]:     # 复合语句在第一句插入if b break
                last_expression_node = abc[3].children[-2]
            else:       # 如果是单行，后面加上了花括号，在就在expression的开始位置插入
                last_expression_node = abc[3]
            indent = get_indent(last_expression_node.start_byte, code)
            res.append((last_expression_node.end_byte, f"\n{indent * ' '}{text(abc[2])};"))
        while_str = f"while({text(abc[1]) if abc[1] else ''})"
        do_str = "do"
        res.append((node.children[0].start_byte, do_str))
        res.append((node.end_byte, while_str))
        if add_bracket:
            res.extend(add_bracket)
        return res
    elif node.type == 'while_statement':
        condition_node = node.children[1]
        return [(node.children[1].end_byte, node.children[0].start_byte),
                (node.children[0].start_byte, "do"),
                (node.children[2].end_byte, f"while{text(condition_node)}")]

def count_do_while(root):
    nodes = match_do_while(root)
    return len(nodes)