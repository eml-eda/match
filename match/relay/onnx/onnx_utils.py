import onnx
import numpy as np

def sanitize_onnx_only_remove(onnx_model: onnx.ModelProto):
    for node in onnx_model.graph.node:
        atts_to_pop = list()
        for idx_att, attribute in enumerate(node.attribute):
            # probably precisions to fix
            if attribute.name=="out_dtype":
                atts_to_pop.append(idx_att)
            if "_" in attribute.name and attribute.name.split("_")[1]=="bits":
                atts_to_pop.append(idx_att)
        for idx_att, att_ref in enumerate(atts_to_pop):
            node.attribute.pop(att_ref - idx_att)

def sanitize_onnx(onnx_model: onnx.ModelProto):
    def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
    ) -> onnx.TensorProto:
        # (TensorProto)
        initializer_tensor = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=tensor_array.shape,
            vals=tensor_array.flatten().tolist(),
        )

        return initializer_tensor

    def numpy_data_type(dtype_type, dtype_bits):
        return np.dtype(f"{dtype_type}{dtype_bits}")

    def onnx_data_type(dtype_type, dtype_bits):
        return onnx.helper.np_dtype_to_tensor_dtype(
            numpy_data_type(dtype_type, dtype_bits)
        )
    
    def get_more_restrictive_npdtype(dtype_list):
        if len(dtype_list)==0:
            return None
        min_itemsize=dtype_list[0].itemsize
        min_int=np.issubdtype(dtype_list[0],np.integer)
        min_unsigned=np.issubdtype(dtype_list[0],np.unsignedinteger)
        for idx_,type_ in enumerate(dtype_list):
            if type_.itemsize<=min_itemsize and idx_>0:
                min_itemsize=type_.itemsize
                if np.issubdtype(type_,np.integer):
                    min_int=True
                    if np.issubdtype(type_,np.unsignedinteger):
                        min_unsigned=True
        dtype_str="u" if min_unsigned else ""
        dtype_str+="int" if min_int else "float"
        dtype_str+=str(min_itemsize*8)
        return np.dtype(dtype_str)
    
    def number_of_inputs_with_bias(op_type):
        if op_type=="Conv":
            return 3
        return 3
    
    nodes_with_new_ops = list()
    nodes_to_append = list()
    output_to_graph_idx= dict()
    nodes_out_types = dict()
    input_names = list()
    input_names_to_idx= dict()
    graph_input_types=dict()
    original_out_to_new_node_version=dict()
    for inp_idx,input_t in enumerate(onnx_model.graph.input):
        input_names.append(input_t.name)
        input_names_to_idx[input_t.name]=inp_idx
        graph_input_types[input_t.name]=onnx.helper.tensor_dtype_to_np_dtype(input_t.type.tensor_type.elem_type)
    for idx_node, node in enumerate(onnx_model.graph.node):
        refactored_idx = int(node.output[0]) + len(nodes_with_new_ops)
        refactored_idx_node = idx_node + len(nodes_with_new_ops)
        weights_bits = -1
        op_bits = -1
        bias_bits = -1
        cast_bits = -1
        atts_to_pop = dict()
        for idx_att, attribute in enumerate(node.attribute):
            if attribute.name=="max" and attribute.f>255:
                attribute.f=255
            if attribute.name=="value":
                nodes_out_types[idx_node]=onnx.helper.tensor_dtype_to_np_dtype(attribute.t.data_type)
            # probably precisions to fix
            if "_" in attribute.name and attribute.name.split("_")[1]=="bits":
                att = attribute.name.split("_")
                if att[0] == "weight":
                    weights_bits = attribute.i
                    atts_to_pop["weight"]=idx_att
                    nodes_out_types[idx_node]=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type("int",weights_bits))
                if hasattr(node, "op_type") and node.op_type.lower() in att[0].lower():
                    op_bits = attribute.i
                    atts_to_pop["op"]=idx_att
                    nodes_out_types[idx_node]=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type("int",op_bits))
                if att[0] == "out":
                    cast_bits = attribute.i
                    atts_to_pop["out"]=idx_att
                    nodes_with_new_ops.append(idx_node)
                    nodes_out_types[idx_node]=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type("int",cast_bits))
                if att[0] == "bias":
                    atts_to_pop["bias"]=idx_att
                    if node.op_type=="Conv":
                        bias_bits = attribute.i
                        #nodes_with_new_ops.append(idx_node)
                        nodes_out_types[idx_node]=onnx.helper.tensor_dtype_to_np_dtype(onnx_data_type("int",bias_bits))
        op_channels = 32
        inputs_types=list()
        for input_idx, node_input in enumerate(node.input):
            if node_input.isnumeric():
                node_input_ref = int(node_input)
                new_node_input_ref = node_input_ref + sum(
                    [node_input_ref > val for val in nodes_with_new_ops]
                )
                node.input[input_idx] = str(new_node_input_ref)
                if node_input not in input_names and onnx_model.graph.node[output_to_graph_idx[node_input_ref]].op_type=="Constant":
                    op_channels_dim_idx = 0
                    op_channels = node_init.dims[op_channels_dim_idx]
                    if op_bits > 0:
                        # const node needs an update
                        onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.raw_data = (
                            np.frombuffer(
                                onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.raw_data,
                                onnx.helper.tensor_dtype_to_np_dtype(
                                    onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type
                                ),
                            )
                            .astype(numpy_data_type("int", op_bits))
                            .tobytes()
                        )
                        onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type = onnx_data_type("int", op_bits)
                        nodes_out_types[output_to_graph_idx[node_input_ref]]=onnx.helper.tensor_dtype_to_np_dtype(onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type)
                    if weights_bits > 0:
                        onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.raw_data = (
                            np.frombuffer(
                                onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.raw_data,
                                onnx.helper.tensor_dtype_to_np_dtype(
                                    onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type
                                ),
                            )
                            .astype(numpy_data_type("int", weights_bits))
                            .tobytes()
                        )
                        onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type = onnx_data_type("int", weights_bits)
                        nodes_out_types[output_to_graph_idx[node_input_ref]]=onnx.helper.tensor_dtype_to_np_dtype(onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type)
                    inputs_types.append({
                        "ndtype":onnx.helper.tensor_dtype_to_np_dtype(onnx_model.graph.node[output_to_graph_idx[node_input_ref]].attribute[0].t.data_type),
                        "type":"constant",
                        "name":node_input_ref,
                        "idx":input_idx,
                        "numeric":True,
                    })
                else:
                    if node_input in input_names:
                        inputs_types.append({
                            "ndtype":graph_input_types[input_t.name],
                            "type":"input",
                            "name":node_input,
                            "idx":input_idx,
                            "numeric":True,
                        })
                    else:
                        inputs_types.append({
                            "ndtype":nodes_out_types[output_to_graph_idx[node_input_ref]],
                            "type":"node",
                            "name":node_input,
                            "idx":input_idx,
                            "numeric":True,
                        })
            else:
                for node_init in onnx_model.graph.initializer:
                    if node_init.name == node_input:
                        if input_idx < number_of_inputs_with_bias(node.op_type)-1:
                            op_channels_dim_idx = 0 if len(node_init.dims) > 1 else 0
                            op_channels = node_init.dims[op_channels_dim_idx]
                            if op_bits > 0:
                                node_init.raw_data = (
                                    np.frombuffer(
                                        node_init.raw_data,
                                        onnx.helper.tensor_dtype_to_np_dtype(
                                            node_init.data_type
                                        ),
                                    )
                                    .astype(numpy_data_type("int", op_bits))
                                    .tobytes()
                                )
                                node_init.data_type = onnx_data_type("int", op_bits)
                            if weights_bits > 0:
                                node_init.raw_data = (
                                    np.frombuffer(
                                        node_init.raw_data,
                                        onnx.helper.tensor_dtype_to_np_dtype(
                                            node_init.data_type
                                        ),
                                    )
                                    .astype(numpy_data_type("int", weights_bits))
                                    .tobytes()
                                )
                                node_init.data_type = onnx_data_type(
                                    "int", weights_bits
                                )
                            inputs_types.append({
                                "ndtype":onnx.helper.tensor_dtype_to_np_dtype(node_init.data_type),
                                "type":"init_constant",
                                "name":node_init.name,
                                "idx":input_idx,
                                "numeric":False,
                            })
                        else:
                            if bias_bits > 0:
                                node_init.raw_data = (
                                    np.frombuffer(
                                        node_init.raw_data,
                                        onnx.helper.tensor_dtype_to_np_dtype(
                                            node_init.data_type
                                        ),
                                    )
                                    .astype(numpy_data_type("int", bias_bits))
                                    .tobytes()
                                )
                                node_init.data_type = onnx_data_type("int", bias_bits)
                                attr_proto=onnx.AttributeProto()
                                attr_proto.name="out_dtype"
                                attr_proto.doc_string = f"int{bias_bits}"
                                attr_proto.type = onnx.AttributeProto.STRING
                                node.attribute.append(attr_proto)
                                #nodes_to_append.append(
                                #    (
                                #        refactored_idx,
                                #        onnx.helper.make_node(
                                #            name=f"{node.name}/bias/cast",
                                #            op_type="Cast",
                                #            inputs=[str(refactored_idx)],
                                #            outputs=[str(refactored_idx+1)],
                                #            to=onnx_data_type("int", bias_bits),
                                #        ),
                                #    )
                                #)
                        break
        
        more_restrictive_dtype = None
        if idx_node in nodes_out_types and bias_bits<0 and cast_bits<0:
            more_restrictive_dtype = nodes_out_types[idx_node]
        else:
            more_restrictive_dtype=get_more_restrictive_npdtype([inp["ndtype"] for inp in inputs_types])
        input_added_nodes=0
        for node_input in inputs_types:
            if node_input["ndtype"]!=more_restrictive_dtype:
                if node_input["type"]=="constant":
                    onnx_model.graph.node[output_to_graph_idx[node_input["name"]]].attribute[0].t.raw_data = (
                        np.frombuffer(
                            onnx_model.graph.node[output_to_graph_idx[node_input["name"]]].attribute[0].t.raw_data,
                            onnx.helper.tensor_dtype_to_np_dtype(
                                onnx_model.graph.node[output_to_graph_idx[node_input["name"]]].attribute[0].t.data_type
                            ),
                        )
                        .astype(more_restrictive_dtype)
                        .tobytes()
                    )
                    onnx_model.graph.node[output_to_graph_idx[node_input["name"]]].attribute[0].t.data_type = onnx.helper.np_dtype_to_tensor_dtype(more_restrictive_dtype)
                    nodes_out_types[output_to_graph_idx[node_input["name"]]]=more_restrictive_dtype
                elif node_input["type"]=="input":
                    onnx_model.graph.input[input_names_to_idx[node_input["name"]]].type.tensor_type.elem_type=onnx.helper.np_dtype_to_tensor_dtype(more_restrictive_dtype)
                elif node_input["type"]=="node":
                    nodes_with_new_ops.append(idx_node-1)
                    nodes_to_append.append(
                        (
                            refactored_idx-1,
                            onnx.helper.make_node(
                                name=f"{node.name}/input_node_{input_added_nodes}/cast",
                                op_type="Cast",
                                inputs=[original_out_to_new_node_version[node_input["name"]]],
                                outputs=[str(refactored_idx+input_added_nodes)],
                                to=onnx.helper.np_dtype_to_tensor_dtype(more_restrictive_dtype),
                            ),
                        )
                    )
                    node.input[node_input["idx"]] = str(refactored_idx+input_added_nodes)
                    input_added_nodes+=1
                elif node_input["type"]=="init_constant":
                    for node_init in onnx_model.graph.initializer:
                        if node_init.name == node_input["name"]:
                            node_init.raw_data = (
                                np.frombuffer(
                                    node_init.raw_data,
                                    onnx.helper.tensor_dtype_to_np_dtype(
                                        node_init.data_type
                                    ),
                                )
                                .astype(more_restrictive_dtype)
                                .tobytes()
                            )
                            node_init.data_type = onnx.helper.np_dtype_to_tensor_dtype(more_restrictive_dtype)
                            break
        
        if bias_bits<0 and cast_bits<0:
            nodes_out_types[idx_node]=more_restrictive_dtype

        if bias_bits > 0 and len(node.input) < number_of_inputs_with_bias(node.op_type):
            bias_arr = np.ones(shape=(op_channels)).astype(
                numpy_data_type("int", bias_bits)
            )
            bias_name = f"{node.name}/bias"
            bias_initializer_tensor = create_initializer_tensor(
                name=bias_name,
                tensor_array=bias_arr,
                data_type=onnx_data_type("int", bias_bits),
            )
            onnx_model.graph.initializer.append(bias_initializer_tensor)
            node.input.append(bias_name)
            attr_proto=onnx.AttributeProto()
            attr_proto.name="out_dtype"
            attr_proto.doc_string = f"int{bias_bits}"
            attr_proto.type = onnx.AttributeProto.STRING
            node.attribute.append(attr_proto)
            # adding a node for the cast of the op to make it acceptable by the bias add precision
            #nodes_to_append.append(
            #    (
            #        refactored_idx,
            #        onnx.helper.make_node(
            #            name=f"{node.name}/bias/cast",
            #            op_type="Cast",
            #            inputs=[str(refactored_idx)],
            #            outputs=[str(refactored_idx+1)],
            #            to=onnx_data_type("int", bias_bits),
            #        ),
            #    )
            #)
        # this node is followed by a cast, so we need to add a new node to the graph
        if cast_bits > 0:
            nodes_to_append.append(
                (
                    refactored_idx,
                    onnx.helper.make_node(
                        name=f"{node.name}/cast",
                        op_type="Cast",
                        inputs=[str(refactored_idx)],
                        outputs=[str(refactored_idx+1)],
                        to=onnx_data_type("int", cast_bits),
                    ),
                )
            )
            #input_added_nodes+=1
            #for inp_idx,node_input in enumerate(node.input):
            #    if node_input.isnumeric():
            #        node_input_ref = int(node_input)
            #        new_node_input_ref = node_input_ref+1
            #        node.input[inp_idx] = str(new_node_input_ref)
        node.output.append(str(refactored_idx + input_added_nodes))
        output_to_graph_idx[int(node.output[0])]=idx_node
        original_out_to_new_node_version[node.output[0]]=str(refactored_idx + input_added_nodes)
        node.output.pop(0)
        for idx_att, att_ref in enumerate(atts_to_pop.values()):
            node.attribute.pop(att_ref - idx_att)
    for idx_node,new_node in enumerate(nodes_to_append):
        onnx_model.graph.node.insert(new_node[0],new_node[1])
    return {"out_types":nodes_out_types,"to_append":nodes_to_append,"out_to_graph_idx":output_to_graph_idx,"new_ops_at":nodes_with_new_ops}