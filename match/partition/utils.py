from tvm import relay

def add_checks_get_first_op(node, op):
    if not isinstance(node, relay.Call):
        print(f"[PATTERN MATCHING-UTILS] Searching op {op} but op {op} is not a call node")
        raise Exception(f"[PATTERN MATCHING-UTILS] Searching op {op} but op {op} is not a call node")
    while node.op.name!=op:
        found = False
        for arg in node.args:
            if isinstance(arg, relay.Call):
                node = arg
                found = True
                break
        if not found:
            print(f"[PATTERN MATCHING-UTILS] Searching op {op} but op {node.op.name} doesn't have any op as args")
            raise Exception(f"[PATTERN MATCHING-UTILS] Searching op {op} but op {node.op.name} doesn't have any op as args")
    return node

def add_checks_get_all_ops(node, op):
    ops = []
    found = True
    while isinstance(node, relay.Call) and found:
        if node.op.name==op:
            ops.append(node)
        found = False
        for arg in node.args:
            if isinstance(arg, relay.Call):
                node = arg
                found = True
                break
    return ops