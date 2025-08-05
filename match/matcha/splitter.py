import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback


class NodeSplitter(DFPatternCallback):
    def __init__(self, match_pattern, pattern_id, matched_patterns_chunks):
        super().__init__(rewrite_once=True)
        self.pattern = match_pattern.pattern()
        self.pattern_id = pattern_id
        self.matched_patterns_chunks = matched_patterns_chunks
        self.device_id = match_pattern.exec_module.id + 1
        self.only_once = 0
    
    def callback(self, pre, post, node_map):
        
        if hasattr(post, "span") and post.span and post.span.source_name.name == "GID":
            gid = post.span.line
            if gid not in self.matched_patterns_chunks or post.span.column > 0:
                return post
        else:
            return post
        
        chunks = self.matched_patterns_chunks[gid]
        if chunks == 0:
            return post
        
        match = []
        def recursive_collect(pattern):
            if pattern in node_map:
                if isinstance(pattern, relay.dataflow_pattern.CallPattern):
                    if node_map[pattern][0] not in match:
                        match.append(node_map[pattern][0])
                    for arg in pattern.args:
                        recursive_collect(arg)
                elif isinstance(pattern, relay.dataflow_pattern.AltPattern):
                    if pattern.left in node_map:
                        recursive_collect(pattern.left)
                    if pattern.right in node_map:
                        recursive_collect(pattern.right)
                
        recursive_collect(self.pattern)
        match = list(reversed(match))
        
        # splitting on H
        
        branch_a = None
        branch_b = None
        inp_fm = match[0].args[0]  # Assuming the first argument is the input feature map
        
        
        try:
            inp_fm_shape = inp_fm.checked_type.shape
        except ValueError:
            return post
        
        print(inp_fm_shape, chunks)
        
        # Find main node
        main_node = None
        for i, node in enumerate(match):
            if hasattr(node, "span") and node.span and node.span.source_name.name == "GID":
                if node.span.column > 0 or node.span.end_line == -1:
                    return post
            if node.op == tvm.ir.Op.get("nn.conv2d"):
                if main_node is not None:
                    raise Exception("Multiple conv2d nodes found in pattern match")
                main_node = node
            
        if not main_node:
            print("No main node found in pattern match")
            return post
        
        # Split input feature map in two branches based on the main node
        if main_node.op == tvm.ir.Op.get("nn.conv2d"):
            print("Pattern main node is conv2d.")
            stride = main_node.attrs.strides[0]
            dilation = main_node.attrs.dilation[0]
            padding = main_node.attrs.padding
            if len(padding) == 2:
                padding_top = padding_bottom = padding[0]
                padding_left = padding_right = padding[1]
            else:
                padding_top, padding_left, padding_bottom, padding_right = padding
            kernel_size = main_node.attrs.kernel_size[0]

            # Tiling dim
            input_shape = main_node.checked_type
            out_h = input_shape.shape[1]  # Assuming NHWC format TODO check data_format
            
            assert main_node.args[0].checked_type.shape == inp_fm_shape, "Input feature map shape does not match main node input shape"

            # ---- First tile: output rows 0 to chunks-1 ----
            inp_fm_height_a = (chunks - 1) * stride + (kernel_size - 1) * dilation + 1 - padding_top
            branch_a = relay.strided_slice(
                inp_fm,
                begin=[0, 0, 0, 0],
                end=[inp_fm_shape[0], inp_fm_height_a, inp_fm_shape[2], inp_fm_shape[3]]
            )
            

            # ---- Second tile: output rows chunks to H-1 ----
            input_start_b = chunks * stride - padding_top
            #input_end_b = (out_h - 1) * stride + (kernel_size - 1) * dilation + 1 - padding_top - padding_bottom
            if inp_fm_shape[1] > input_start_b and chunks != out_h:
                branch_b = relay.strided_slice(
                    inp_fm,
                    begin=[0, input_start_b, 0, 0],
                    end=[inp_fm_shape[0], inp_fm_shape[1], inp_fm_shape[2], inp_fm_shape[3]]
                    )
                
    
        # Split ops
        for i, node in enumerate(match):
            if node.op == tvm.ir.Op.get("nn.conv2d"):
                conv_attrs = {key: node.attrs[key] for key in node.attrs.keys()}
                padding = tuple(conv_attrs.get('padding', (0, 0)))
                
                if isinstance(padding, (list, tuple)) and len(padding) == 4:
                    padding_conv_top = (padding[0], padding[1], 0, padding[3])
                    padding_conv_bottom = (0, padding[1], padding[2], padding[3])
                elif len(padding) == 2:
                    padding_conv_top = (padding[0], padding[1], 0, padding[1])
                    padding_conv_bottom = (0, padding[1], padding[0], padding[1])
                else:
                    padding_conv_top = (padding[0], padding[0], 0, padding[0])
                    padding_conv_bottom = (0, padding[0], padding[0], padding[0])
                
                if branch_b: # if not branch_b -> chunks == out_h so no need to modify padding
                    conv_attrs['padding']  = padding_conv_top
                
                node_ = relay.nn.conv2d(branch_a, node.args[1], **conv_attrs) # Workaround to modify padding but keep span... TVM :)
                branch_a = tvm.relay.Call(
                    node_.op,
                    node_.args,
                    attrs=node_.attrs,
                    type_args=node.type_args,
                    span=tvm.ir.Span(node.span.source_name, node.span.line, 0, self.device_id, 0)
                )
                
                if branch_b:
                    conv_attrs['padding'] = padding_conv_bottom
                    node_ = relay.nn.conv2d(branch_b, node.args[1], **conv_attrs)
                    branch_b = tvm.relay.Call(
                        node_.op,
                        node_.args,
                        attrs=node_.attrs,
                        type_args=node.type_args,
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, self.device_id, 0)
                    )
            else:
                branch_a = tvm.relay.Call(
                    node.op,
                    [branch_a] + list(node.args[1:]),
                    attrs=node.attrs,
                    type_args=node.type_args,
                    span=tvm.ir.Span(node.span.source_name, node.span.line, 0, self.device_id, 0)
                )
                if branch_b:
                    branch_b = tvm.relay.Call(
                        node.op,
                        [branch_b] + list(node.args[1:]),
                        attrs=node.attrs,
                        type_args=node.type_args,
                        span=tvm.ir.Span(node.span.source_name, node.span.line, 0, -1, 0)
                    )
        
        if branch_b:
            self.only_once += 1
            #if self.only_once != 4:
            #    return post
            return relay.concatenate([branch_a, branch_b], axis=1)        

        print("missing other branch")
        #return post
        return branch_a