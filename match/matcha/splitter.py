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
        
        # Find main node
        main_node = None
        for i, node in enumerate(match):
            if hasattr(node, "span") and node.span and node.span.source_name.name == "GID":
                if node.span.column > 0: # or node.span.end_line == -1:
                    return post
            if node.op in [tvm.ir.Op.get("nn.conv2d"), tvm.ir.Op.get("nn.dense")]:
                if main_node is not None:
                    raise Exception("Multiple main nodes found in pattern match")
                main_node = node
            
        if not main_node:
            print("No main node found in pattern match")
            return post
        
        # Split match in two branches based on the main node (conv, dense, etc)
        if main_node.op == tvm.ir.Op.get("nn.conv2d"):
            return self.split_conv2d(pre, post, match, main_node, chunks)
        elif main_node.op == tvm.ir.Op.get("nn.dense"):
            return self.split_dense(pre, post, match, main_node, chunks)
        else:
            print(f"Pattern main node ({main_node.op}) is not supported for splitting.")
            return post
        
    
    @staticmethod
    def conv_height_slice_for_output_range(H_in, y_start, y_end_incl, k_h, s_h, d_h, pad_top, pad_bottom):
        # Calculate the input height slice needed and vertical paddings to produce output rows y_start to y_end_incl
        # considering kernel size k_h, stride s_h, dilation d_h, and padding (pad_top, pad_bottom).
        
        in_padded_start = y_start * s_h
        in_padded_end = y_end_incl * s_h + (k_h - 1) * d_h   # inclusive
        in_unpadded_start = in_padded_start - pad_top
        in_unpadded_end_excl = in_padded_end - pad_top + 1

        slice_start = max(0, in_unpadded_start)
        slice_end_excl = min(H_in, in_unpadded_end_excl)

        pad_before = max(0, -in_unpadded_start)
        pad_after = max(0, in_unpadded_end_excl - H_in)

        return slice_start, slice_end_excl, pad_before, pad_after
        
        
    
    def split_conv2d(self, pre, post, match, main_node, chunks):
        # Splitting on output height dimension (H)
        print("Pattern main node is conv2d.")
        
        branch_a = None
        branch_b = None
        
        
        inp_fm = match[0].args[0]
        try:
            inp_fm_shape = inp_fm.checked_type.shape
        except ValueError:
            print("  Cannot get input feature map shape, skipping split.")
            return post
        
        print("   Input fm shape:", inp_fm_shape, " - Main node out shape:", main_node.checked_type.shape)
        print("   Chunks:", chunks, "- Output height:", main_node.checked_type.shape[1])
        print("   Strides:", main_node.attrs.strides, "- Dilation:", main_node.attrs.dilation, 
               "- Padding:", main_node.attrs.padding, "- Kernel size:", main_node.attrs.kernel_size)

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
        out_shape = main_node.checked_type
        out_h = out_shape.shape[1]  # Assuming NHWC format TODO check data_format
        in_h = inp_fm_shape[1]
        
        assert main_node.args[0].checked_type.shape == inp_fm_shape, "Input feature map shape does not match main node input shape"

        # ---- First tile: output rows 0 to chunks-1 - calculate receptive field slice ----
        inp_slice_a_start, inp_slice_a_end, pad_top_a, pad_bottom_a = self.conv_height_slice_for_output_range(
            in_h, 0, chunks - 1, kernel_size, stride, dilation, padding_top, padding_bottom)
        
        if inp_slice_a_start == 0 and inp_slice_a_end == out_h:
            branch_a = inp_fm
        else:
            branch_a_begin = [0, 0, inp_slice_a_start, 0]
            branch_a_end = [inp_fm_shape[0], inp_slice_a_end, inp_fm_shape[2], inp_fm_shape[3]]
            branch_a = relay.strided_slice(inp_fm, begin = branch_a_begin, end = branch_a_end)
            print("  branch_a slice", branch_a_begin, branch_a_end, "pad_top_a", pad_top_a, "pad_bottom_a", pad_bottom_a)
            
        # ---- Second tile: output rows chunks to H-1 - calculate receptive field slice ----
        inp_slice_b_start, inp_slice_b_end, pad_top_b, pad_bottom_b = self.conv_height_slice_for_output_range(
            in_h, chunks, out_h - 1, kernel_size, stride, dilation, padding_top, padding_bottom)
        
        if chunks != out_h:
            if inp_slice_b_start == 0 and inp_slice_b_end == out_h:
                branch_b = inp_fm
            else:
                branch_b_begin = [0, inp_slice_b_start, 0, 0]
                branch_b_end = [inp_fm_shape[0], inp_slice_b_end, inp_fm_shape[2], inp_fm_shape[3]]
                branch_b = relay.strided_slice(inp_fm, begin = branch_b_begin, end = branch_b_end)
                print("  branch_b slice", branch_b_begin, branch_b_end, "pad_top_b", pad_top_b, "pad_bottom_b", pad_bottom_b)
    
        # Split ops
        for i, node in enumerate(match):
            if node.op == tvm.ir.Op.get("nn.conv2d"):
                conv_attrs = {key: node.attrs[key] for key in node.attrs.keys()}
                
                conv_attrs['padding'] = (pad_top_a, padding_left, pad_bottom_a, padding_right)
                
                node_ = relay.nn.conv2d(branch_a, node.args[1], **conv_attrs) # Workaround to modify padding but keep span... TVM :)
                branch_a = tvm.relay.Call(
                    node_.op,
                    node_.args,
                    attrs=node_.attrs,
                    type_args=node.type_args,
                    span=tvm.ir.Span(node.span.source_name, node.span.line, 0, self.device_id, 0)
                )
                
                if branch_b:
                    conv_attrs['padding'] = (pad_top_b, padding_left, pad_bottom_b, padding_right)
                    node_ = relay.nn.conv2d(branch_b, node.args[1], **conv_attrs)
                    branch_b = tvm.relay.Call(
                        node_.op,
                        node_.args,
                        attrs=node_.attrs,
                        type_args=node.type_args,
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, 0, 0)
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
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, 0, 0)
                    )
        
        if branch_b:
            self.only_once += 1
            #if self.only_once != 4:
            #    return post
            return relay.concatenate([branch_a, branch_b], axis=1)        

        print("  Missing other branch.")
        #return post
        return branch_a


    def split_dense(self, pre, post, match, main_node, chunks):
        # Splitting on output feature/neuron dimension
        print("Pattern main node is dense.")

        branch_a = None
        branch_b = None
        
        inp_fm = match[0].args[0]
        weight_shape = main_node.args[1].checked_type.shape
        
        # Tiling dim
        print("   Chunks:", chunks, "- Output features:", weight_shape[0])
        print("   Input fm shape:", inp_fm.checked_type.shape)
        print("   Weight shape:", weight_shape)
        print("   Main node out shape:", main_node.checked_type.shape)

        # Split ops
        branch_a = inp_fm
        branch_b = inp_fm if weight_shape[0] != chunks else None
        
         # Split ops
        
        for i, node in enumerate(match):
            if node.op == tvm.ir.Op.get("nn.dense"):
                
                if weight_shape[0] != chunks:
                    weight_a = relay.strided_slice(node.args[1], begin=[0]*len(weight_shape), end=[chunks, *weight_shape[1:]])
                else:
                    weight_a = node.args[1]
                    
                node_ = relay.nn.dense(branch_a, weight_a)
                branch_a = tvm.relay.Call(
                    node_.op,
                    node_.args,
                    attrs=node_.attrs,
                    type_args=node.type_args,
                    span=tvm.ir.Span(node.span.source_name, node.span.line, 0, self.device_id, 0)
                )
                
                if branch_b:
                    weight_b = relay.strided_slice(node.args[1], begin=[chunks] + [0]*len(weight_shape[1:]), end=[*weight_shape])
                    node_ = relay.nn.dense(branch_b, weight_b)
                    branch_b = tvm.relay.Call(
                        node_.op,
                        node_.args,
                        attrs=node_.attrs,
                        type_args=node.type_args,
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, 0, 0)
                    )  
                    
            elif node.op == tvm.ir.Op.get("add"): 
                print("   Bias shape", node.args[1].checked_type.shape)
                
                if weight_shape[0] != chunks:
                    weight_a = relay.strided_slice(
                        node.args[1],
                        begin=[0]*len(node.args[1].checked_type.shape),
                        end=list(node.args[1].checked_type.shape[:-1]) + [chunks]
                    )
                else:
                    weight_a = node.args[1]
                node_ = relay.add(branch_a, weight_a)
                branch_a = tvm.relay.Call(
                    node_.op,
                    node_.args,
                    attrs=node_.attrs,
                    type_args=node.type_args,
                    span=tvm.ir.Span(node.span.source_name, node.span.line, 0, self.device_id, 0)
                )
                if branch_b:
                    weight_b = relay.strided_slice(
                        node.args[1],
                        begin=[0]*len(node.args[1].checked_type.shape[:-1]) + [chunks],
                        end=list(node.args[1].checked_type.shape)
                    )
                    node_ = relay.add(branch_b, weight_b)
                    branch_b = tvm.relay.Call(
                        node_.op,
                        node_.args,
                        attrs=node_.attrs,
                        type_args=node.type_args,
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, 0, 0)
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
                        span=tvm.ir.Span(node.span.source_name, node.span.line, -1, 0, 0)
                    )
        
        if branch_b:
            self.only_once += 1
            #if self.only_once != 4:
            #    return post
            return relay.concatenate([branch_a, branch_b], axis=-1)        

        print("  Missing other branch.")
        #return post
        return branch_a