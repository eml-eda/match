

from floats import create_fp_add_ex, create_fp_conv_ex, create_fp_dense_ex, create_fp_globalavgpool_ex
from mini import create_add_fp_vit, create_addlayernorm_fp_vit, create_attention_fp_vit, create_classifier_fp_vit, create_densegelu_fp_vit, create_densemlp_fp_vit, create_fp_conv_vit, create_fp_layer_norm_vit, create_fp_proj_vit

VIT_MAPPER ={
    "conv":create_fp_conv_vit,
    "layernorm":create_fp_layer_norm_vit,
    "proj":create_fp_proj_vit,
    "attention":create_attention_fp_vit,
    "addlayernorm":create_addlayernorm_fp_vit,
    "densegelu":create_densegelu_fp_vit,
    "densemlp":create_densemlp_fp_vit,
    "add":create_add_fp_vit,
    "classifier":create_classifier_fp_vit,
}

VIT_NODES = [
    "vit_fp/conv",
    "vit_fp/layernorm",
    "vit_fp/proj/projections-1", # 1 PROJS AT A TIME
    "vit_fp/attention/out_ch-198", 
    "vit_fp/addlayernorm",
    "vit_fp/densegelu/projections-1/out_ch-198", # 1 PROJS AT A TIME 
    "vit_fp/densemlp/projections-1/out_ch-198", # 1 PROJS AT A TIME 
    "vit_fp/add", 
    "vit_fp/classifier",
]

RESNET_18_MAPPER = {
    "conv":create_fp_conv_ex,
    "dense":create_fp_dense_ex,
    "global_avgpool":create_fp_globalavgpool_ex,
    "add":create_fp_add_ex,
}

RESNET_18_NODES = [
    # tail
    "resnet_18/conv/out_ch-64/inp_ch-3/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
    # basic block no exp conv 1
    "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
    # basic block no exp conv 2
    "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
    # basic block no exp conv 3
    "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
    # basic block no exp conv 4
    "resnet_18/conv/out_ch-64/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-1x1/padding-1x1x1x1",
    # basic block expand conv 1
    "resnet_18/conv/out_ch-128/inp_ch-64/fil_shape-3x3/inp_shape-32x32/strides-2x2/padding-1x1x1x1",
    # basic block expand conv 2
    "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
    # basic block expand shortcut
    "resnet_18/conv/out_ch-128/inp_ch-64/fil_shape-1x1/inp_shape-32x32/strides-2x2/padding-0x0x0x0",
    # basic block expand conv 3
    "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
    # basic block expand conv 4
    "resnet_18/conv/out_ch-128/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-1x1/padding-1x1x1x1",
    # basic block expand 2 conv 1
    "resnet_18/conv/out_ch-256/inp_ch-128/fil_shape-3x3/inp_shape-16x16/strides-2x2/padding-1x1x1x1",
    # basic block expand 2 conv 2
    "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
    # basic block expand 2 shortcut
    "resnet_18/conv/out_ch-256/inp_ch-128/fil_shape-1x1/inp_shape-16x16/strides-2x2/padding-0x0x0x0",
    # basic block expand 2 conv 3
    "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
    # basic block expand 2 conv 4
    "resnet_18/conv/out_ch-256/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-1x1/padding-1x1x1x1",
    # basic block expand 3 conv 1
    "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-3x3/inp_shape-8x8/strides-2x2/padding-1x1x1x1",
    # basic block expand 3 conv 2
    "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
    # basic block expand 3 shortcut
    "resnet_18/conv/out_ch-512/inp_ch-256/fil_shape-1x1/inp_shape-8x8/strides-2x2/padding-0x0x0x0",
    # basic block expand 3 conv 3
    "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
    # basic block expand 3 conv 4
    "resnet_18/conv/out_ch-512/inp_ch-512/fil_shape-3x3/inp_shape-4x4/strides-1x1/padding-1x1x1x1",
    # head globalavgpool
    "resnet_18/global_avgpool/inp_shape-512x4x4",
    # head dense
    "resnet_18/dense/inp_features-512/out_features-100",
    # add layers
    "resnet_18/add/inp_shape-64x32x32",
    "resnet_18/add/inp_shape-128x16x16",
    "resnet_18/add/inp_shape-256x8x8",
    "resnet_18/add/inp_shape-512x4x4",
]