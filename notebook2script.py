import json, sys
# from exceptions import ArgumentError

def read_notebook(f):
    with open(f, "r", encoding="utf-8") as fp:
        obj = json.load(fp)
    return obj
    
def write_script(f, script):
    with open(f, "wt", encoding="utf-8") as fp:
        fp.write(script)
        
def cells(obj):
    return obj["cells"]

def code_cells(cells):
    return [cell for cell in cells if cell['cell_type'] == "code"]

def sources(cells):
    return [cell['source'] for cell in cells]

def codes(sources):
    codes = ["".join(source) for source in sources]
    return [code + "\n" for code in codes]


#only definition code must be transformed to py script (func, class, imports, comments...)
def is_test(inst):
    pass

def is_import(inst):
    pass

def is_magic(inst):
    pass

def is_function(inst):
    pass

def is_class(inst):
    pass

def is_help(inst):
    pass

def is_comment(inst):
    pass

def is_pass(inst):
    return True

def codes_list(f):
    return [code for code in codes(sources(code_cells(cells(read_notebook(f))))) if is_pass(code) == True]

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        i = sys.argv[1]
    else:
        raise Exception
        
    if len(sys.argv) > 2:
        o = sys.argv[2]
    else:
        raise Exception    
    
    script_list = codes_list(i)
    script = "".join(script_list)
    write_script(o, script)
    print("converting notebook {} to script and saving in {}.".format(i, o))
    