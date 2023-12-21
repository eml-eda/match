class PartitioningPattern:
    def __init__(self,name,pattern,additional_checks):
        self.name=name
        self.pattern=pattern
        self.additional_checks=additional_checks


def conv2d_bias_requant_pattern(name="default_conv2d_bias_requant",
                                supported_fs_and_expected_padding=[(1,[0]),(3,[1]),(5,[2]),(7,[3])],
                                only_equal_padding_supported=True,
                                supported_strides=[[1, 1], [2, 2]],
                                supported_dilations=[[1,1]],
                                supported_kernel_layout=["OIHW"],
                                supported_data_layouts=["NCHW"],
                                supported_groups=[1]):
    def _requant_pattern(prev_op):
        """Add requant pattern (right_shift -> clip -> cast) to prev_op"""
        right_shift = is_op("right_shift")(prev_op, is_constant())
        clip = is_op("clip")(right_shift)
        cast = is_op("cast")(clip).has_attr({"dtype": "uint8"})
        return cast


    def _biasadd_requant_pattern(linear_op):
        """Add pattern bias_add-requant to linear_op"""

        bias_add = is_op("nn.bias_add")(linear_op, wildcard()) | is_op("add")(linear_op, wildcard())
        return _requant_pattern(bias_add)


    def conv2d_pattern():
        """Create pattern for conv2D with optional fused relu."""
        #breakpoint()
        conv2d = is_op("nn.conv2d")(
                wildcard(), wildcard()
        )
        return _biasadd_requant_pattern(conv2d)

    def check_conv2d(pattern):
        """Check if the Conv2D is supported by the soma dory accelerator"""
        #breakpoint()
        conv2d = _check_biasadd_requant(pattern)
        if conv2d is None:
            return False

        num_output_channels = conv2d.args[1].data.shape[0]

        def is_conv2d_attr_value_supported(attrs, name, supported_values):
            attr = attrs[name]

            if isinstance(attr, tvm.ir.container.Array):
                attr = list(attr)

            if attr not in supported_values:
                logger.warning(f"Expected nn.conv2d {name} to be one of {supported_values}, but got {attr}. " +\
                                "Acceleration for this op is not supported.")
                return False

            return True

        def is_kernel_shape_supported(attrs):
            kernel_size = list(attrs["kernel_size"])
            kernel_h = kernel_size[0]
            kernel_w = kernel_size[1]
            supported_kernel_shapes=[v[1] for v in supported_fs_and_expected_padding]
            if (kernel_h not in supported_kernel_shapes) or (kernel_w not in supported_kernel_shapes):
                logger.warning(f"Expected nn.conv2d kernel width and height to be one of {supported_kernels}, " +\
                            f"but got {kernel_size}. " +\
                                "Acceleration for this op is not supported.")
                return False

        def is_padding_supported(attrs):
            # In topi, padding is [padt, padl, padb, padr]
            padding = list(attrs["padding"])
            # Only support equal left-right and top-bottom padding
            supported_padding_values=[v[2] for v in supported_fs_and_expected_padding]
            if any([pval not in supported_padding_values for pval in padding]):
                logger.warning(f"Expected nn.conv2d padding to be like {supported_padding_values}" +\
                                "Acceleration for this op is not supported.")
                return False
            if only_equal_padding_supported:
                if (padding[0] != padding[2]) or (padding[1] != padding[3]):
                    logger.warning(f"Expected equal top and bottom padding, and equal left and right padding," +\
                            f"but got {[padding[0], padding[2]]} and {[padding[1], padding[3]]}, respectively. " +\
                                "Acceleration for this op is not supported.")
                    return False

        def is_padding_as_fs_expected(attrs):
            kernel_size = list(attrs["kernel_size"])
            kernel_h = kernel_size[0]
            kernel_w = kernel_size[1]
            expected_pd_h=0
            expected_pd_w=0
            for fs_pad in supported_fs_and_expected_padding:
                if fs_pad[0]==kernel_h:
                    expected_pad_h=fs_pad[1]
                if fs_pad[0]==kernel_w:
                    expected_pad_w=fs_pad[1]
            # In topi, padding is [padt, padl, padb, padr]
            padding = list(attrs["padding"])
            # Only support output with same output dimension on accelerator
            if padding[0] not in expected_pad_h or padding[2] not in expected_pad_h or padding[1] not in expected_pad_w or padding[3] not in expected_pad_w:
                logger.warning(f"Accelerator only supports 'SAME' padding. " +\
                            f"Expected nn.conv2d with kernel size {kernel_size} to have padding {expected_pad_h} for the height and {expected_pad_w} for the width, " +\
                            f"but got {padding}.")
                return False

            return True


        # check conv2d attributes
        if (not is_kernel_shape_supported(conv2d.attrs)
            or not is_padding_supported(conv2d.attrs)
            or not is_padding_as_fs_expected(conv2d.attrs)
            or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', supported_strides)
            or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', supported_dilations)
            or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', supported_groups+[num_output_channels])
            or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', supported_kernel_layout)
            or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', supported_data_layouts)):

            return False

        return True
    return PartitioningPattern(name=name,pattern=conv2d_pattern,additional_checks=check_conv2d)