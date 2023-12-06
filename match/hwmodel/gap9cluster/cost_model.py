from zigzag.classes.cost_model.specialized_latency_cost_model import SpecializedLatencyCostModel
from zigzag.classes.cost_model.specialized_latency_cost_model import default_ideal_temporal_cycles,default_ideal_cycles
from zigzag.classes.opt.temporal.loma.memory_allocator import MemoryAllocator
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from math import prod,ceil,floor

def calc_computational_cycles(spec_cost_model: SpecializedLatencyCostModel):
    def _floor(ch, N):
        return floor((ch + N - 1) / N)
    spatial_mapping_sizes = prod(
        [
            dim[1]
            for (
                key,
                dim,
            ) in spec_cost_model.cost_model.temporal_mapping.layer_node.user_spatial_mapping.items()
        ]
    )
    lsize = spec_cost_model.cost_model.temporal_mapping.layer_node.loop_dim_size
    lpsize = spec_cost_model.cost_model.temporal_mapping.layer_node.pr_loop_dim_size
    latency=0
    #breakpoint()
    ch_in = spec_cost_model.cost_model.dmaconfstruct[spec_cost_model.cost_model.layer.input_operands[0]]['len_1d_copy']
    ch_out = spec_cost_model.cost_model.dmaconfstruct['O']['len_1d_copy']
    #print(f"\n\nDma conf struct {spec_cost_model.cost_model.dmaconfstruct}\n\n")
    kernel_size_x = lsize['FX']
    kernel_size_y = lsize['FY']
    #input_shape=[lsize['B'],spec_cost_model.cost_model.dmaconfstruct['I']['len_1d_copy'],spec_cost_model.cost_model.dmaconfstruct['I']['num_2d_copies'],spec_cost_model.cost_model.dmaconfstruct['I']['num_1d_copies']]
    output_shape=[lsize['B'],spec_cost_model.cost_model.dmaconfstruct['O']['len_1d_copy'],spec_cost_model.cost_model.dmaconfstruct['O']['num_2d_copies'],spec_cost_model.cost_model.dmaconfstruct['O']['num_1d_copies']]
    #breakpoint()
    strides=[spec_cost_model.cost_model.layer.layer_attrs["attrs"]["strides"]['IY'],spec_cost_model.cost_model.layer.layer_attrs["attrs"]["strides"]['IX']]
    #print(f"\n\nCh in {ch_in} ch out {ch_out} i shape {input_shape} strides {strides}")
    if spec_cost_model.cost_model.layer.layer_attrs['operator_type']=="conv_2d":
        iterations = _floor(int(output_shape[2]*strides[0]), 8)* _floor(int(output_shape[3]*strides[1]), 2) * _floor(int(ch_out), 4)
        im2col = kernel_size_x * kernel_size_y * ch_in * 2
        matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
        latency += iterations * (im2col + matmul)
    elif spec_cost_model.cost_model.layer.layer_attrs['operator_type']=='depthwise_conv_2d':
        # 1 MAC/cycle
        latency = 4 * _floor(ch_out, 8)  * _floor(output_shape[3]*strides[1],4) * kernel_size_x * kernel_size_y * int(output_shape[2]*strides[0])
    elif spec_cost_model.cost_model.layer.layer_attrs['operator_type']=='dense':
        latency += _floor(ch_in, 2) * _floor(ch_out, 4)
    else:
        latency += _floor(ch_in, 2) * _floor(ch_out, 4)
    #print(f"\n\nLatency is{latency}\n\n")
    # save in latency 0
    #latency=1
    spec_cost_model.cost_model.mapping_int.temporal_mapping.total_cycle=latency*spec_cost_model.cost_model.multiplicity_l2['O']
    spec_cost_model.cost_model.ideal_temporal_cycle = (
        spec_cost_model.cost_model.mapping_int.temporal_mapping.total_cycle
    )
    spec_cost_model.cost_model.ideal_cycle = spec_cost_model.cost_model.ideal_temporal_cycle
    spec_cost_model.cost_model.single_comp = spec_cost_model.cost_model.ideal_temporal_cycle//spec_cost_model.cost_model.multiplicity_l2['O'] + 250
    #print(f"\n\nSingle comp is {spec_cost_model.cost_model.single_comp} for {spec_cost_model.cost_model.multiplicity_l2['O']} so total comp is {spec_cost_model.cost_model.ideal_temporal_cycle}\n\n")

def calc_multiplicity_l2_and_transfer_overheads(spec_cost_model: SpecializedLatencyCostModel):
    spec_cost_model.cost_model.multiplicity_l2 = {
        key: prod([v[1] for v in val[len(val) - 1]])
        for (key, val) in spec_cost_model.cost_model.temporal_mapping.mapping_dic_stationary.items()
    }
    # diana contrib
    def layout_sorted(operand):
        if operand in ['I','X','Y']:
            return ["OY","OX",'C' if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 'K']
        elif operand=="W":
            return ["K","FY","FX","C"]
        else:
            return ["OY","OX","K"]
    spec_cost_model.cost_model.multiplicity_l2 = {
        key: (max(spec_cost_model.cost_model.multiplicity_l2.values()) if key == "O" else val)
        for key, val in spec_cost_model.cost_model.multiplicity_l2.items()
    }
    tmap = spec_cost_model.cost_model.temporal_mapping.mapping_dic_stationary
    lsize = spec_cost_model.cost_model.temporal_mapping.layer_node.loop_dim_size
    lpsize = spec_cost_model.cost_model.temporal_mapping.layer_node.pr_loop_dim_size
    relmap = {
        key: {
            "r": layout_sorted(key),
            "ir": val["ir"],
        }
        for (
            key,
            val,
        ) in spec_cost_model.cost_model.temporal_mapping.layer_node.operand_loop_dim.items()
    }
    multiplicity_rel_L2 = {
        operand: {
            reldim: prod(
                [val[1] for val in tmap[operand][len(tmap[operand]) - 1] if val[0] == reldim]
            )
            for reldim in relmap[operand]["r"]
        }
        for operand in spec_cost_model.cost_model.temporal_mapping.operand_list
    }
    #breakpoint()
    for comm in set(relmap["O"]["r"]).intersection(
        set(
            [
                val
                for key, dictval in relmap.items()
                if key not in ["O"]
                for val in dictval["r"]
            ]
        )
    ):
        multiplicity_rel_L2["O"][comm] = max(
            [dictval[comm] for key, dictval in multiplicity_rel_L2.items() if comm in dictval]
        )

    def get_stride_2_op(operand):
        if operand in ['I','X','Y']:
            return lsize['C' if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 'K']*lpsize['IX']
        elif operand=='W':
            return lsize['C']*lsize['FY']*lsize['FX']
        elif operand=='O':
            return lsize['K']*lsize['OX']
    def get_stride_1_op(operand):
        if operand in ['I','X','Y']:
            return lsize['C' if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 'K']
        elif operand=='W':
            return lsize['C']
        elif operand=='O':
            return lsize['K']
    def get_num_2d_copies_op(operand):
        if operand in ['I','X','Y']:
            return lpsize["IY"]//multiplicity_rel_L2[operand]["OY"]
        elif operand=='W':
            return lsize['K']//multiplicity_rel_L2["W"]["K"] if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 1
        elif operand=='O':
            return lsize['OY']//multiplicity_rel_L2["O"]["OY"]
    def get_num_1d_copies_op(operand):
        if operand in ['I','X','Y']:
            return lpsize["IX"]//multiplicity_rel_L2[operand]["OX"]
        elif operand=='W':
            return lsize['FY']*lsize['FX']*lsize['C'] if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 1
        elif operand=='O':
            return lsize['OX']//multiplicity_rel_L2["O"]["OX"]
    def get_len_1d_copy_op(operand):
        if operand in ['I','X','Y']:
            return lsize['C' if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 'K']//multiplicity_rel_L2[operand]['C' if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else 'K']
        elif operand=='W':
            return lsize['C'] if spec_cost_model.cost_model.layer.layer_attrs['operator_type']!='depthwise_conv_2d' else (lsize['K']//multiplicity_rel_L2["W"]["K"])*lsize['FY']*lsize['FX']
        elif operand=='O':
            return lsize['K']//multiplicity_rel_L2["O"]["K"]
    
    spec_cost_model.cost_model.dmaconfstruct={
        operand:{
            'hwc_to_cwh':operand=='I' and spec_cost_model.cost_model.layer.layer_attrs['operator_type']=='depthwise_conv_2d',
            'stride_2d':get_stride_2_op(operand),
            'stride_1d':get_stride_1_op(operand),
            'num_2d_copies':get_num_2d_copies_op(operand),
            'num_1d_copies':get_num_1d_copies_op(operand),
            'len_1d_copy':get_len_1d_copy_op(operand),
        } for operand in spec_cost_model.cost_model.temporal_mapping.operand_list
    }
    def calc_overhead(operand):
        if spec_cost_model.cost_model.dmaconfstruct[operand]['hwc_to_cwh']:
            return (27*spec_cost_model.cost_model.dmaconfstruct[operand]['len_1d_copy'])+1000
        elif spec_cost_model.cost_model.dmaconfstruct[operand]['num_2d_copies']==1 and spec_cost_model.cost_model.dmaconfstruct[operand]['num_1d_copies']==1:
            return 100+300
        else:
            return (27*spec_cost_model.cost_model.dmaconfstruct[operand]['num_2d_copies'])+300
        
    spec_cost_model.cost_model.overhead_per_op={operand:calc_overhead(operand) for operand in spec_cost_model.cost_model.temporal_mapping.operand_list}

    def calc_total_transfer_cost_per_op(operand):
        if operand=='O':
            return spec_cost_model.cost_model.data_offloading_cc_pair_combined[0]+spec_cost_model.cost_model.overhead_per_op['O']
        else:
            #if operand=='I' and spec_cost_model.cost_model.layer.layer_attrs['operator_type']=='depthwise_conv_2d':
                #return spec_cost_model.cost_model.dmaconfstruct['I']['len_1d_copy']*(2*spec_cost_model.cost_model.dmaconfstruct['I']['num_2d_copies']*spec_cost_model.cost_model.dmaconfstruct['I']['num_1d_copies'])+spec_cost_model.cost_model.overhead_per_op['I']
            #else:
            return (spec_cost_model.cost_model.data_loading_cc_pair_combined_per_op[operand][0]*(2 if spec_cost_model.cost_model.dmaconfstruct[operand]['hwc_to_cwh'] else 1))+spec_cost_model.cost_model.overhead_per_op[operand]
    spec_cost_model.cost_model.multiplicity_rel_L2=multiplicity_rel_L2
    spec_cost_model.cost_model.relmap=relmap
    spec_cost_model.cost_model.transfer_cost_per_op={operand:calc_total_transfer_cost_per_op(operand) for operand in spec_cost_model.cost_model.temporal_mapping.operand_list}
    


def calc_loading_and_latency1(spec_cost_model: SpecializedLatencyCostModel):
    #spec_cost_model.cost_model.total_loading_cycles = sum(
    #    [spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O']
    #)
    cyc=0
    cyc_1=0
    mults=sorted(set(spec_cost_model.cost_model.multiplicity_l2.values()))
    prev_mult_=0
    for idx,mult_ in enumerate(mults):
        if idx==0:
            cyc+=max([0]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
            cyc_1+=max([0]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
            prev_mult_=1
        cyc+=(mult_-prev_mult_)*max([spec_cost_model.cost_model.single_comp]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
        cyc_1+=(mult_-prev_mult_)*max([0]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
        prev_mult_=mult_
    spec_cost_model.cost_model.cyc_1=cyc_1
    spec_cost_model.cost_model.latency_total0=spec_cost_model.cost_model.single_comp*spec_cost_model.cost_model.multiplicity_l2['O']
    spec_cost_model.cost_model.latency_total1=cyc+spec_cost_model.cost_model.single_comp


def calc_offloading_and_latency2(spec_cost_model: SpecializedLatencyCostModel):
    cme_execution=True
    if cme_execution:
        cyc=0
        cyc_2=0
        mults=sorted(set(spec_cost_model.cost_model.multiplicity_l2.values()))
        prev_mult_=0
        for idx,mult_ in enumerate(mults):
            if idx==0:
                cyc+=max([0]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
                cyc_2+=max([0]+[spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if operand!='O' and spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])
                prev_mult_=1
            cyc+=(mult_-prev_mult_)*max([spec_cost_model.cost_model.single_comp,max([spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])+300])
            cyc_2+=(mult_-prev_mult_)*max([0,max([spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list if spec_cost_model.cost_model.multiplicity_l2[operand]>=mult_])+300])
            prev_mult_=mult_
        spec_cost_model.cost_model.cyc_2=cyc_2+spec_cost_model.cost_model.transfer_cost_per_op['O']
        spec_cost_model.cost_model.latency_total2=cyc+spec_cost_model.cost_model.single_comp+spec_cost_model.cost_model.transfer_cost_per_op['O']
    else:
        spec_cost_model.cost_model.latency_total2=spec_cost_model.cost_model.latency_total0+sum([spec_cost_model.cost_model.transfer_cost_per_op[operand] for operand in spec_cost_model.cost_model.temporal_mapping.operand_list] )
    spec_cost_model.cost_model.energy_total=0
    #print(f"\n\nLatency for {spec_cost_model.cost_model.temporal_mapping.mapping_dic_stationary} was Comp {spec_cost_model.cost_model.latency_total0} total {spec_cost_model.cost_model.latency_total2}\n\n")


def buffer_size_and_db_mem_func(mem_all: MemoryAllocator, mem_instance: MemoryInstance):
    buff_mem=0
    db_support=False
    if mem_instance.name=='shared_l1':
        # buffer for the cores of the accelerator (weights dimensions)
        if mem_all.layer.layer_attrs['operator_type']=='conv_2d' and (mem_all.layer.loop_dim_size['FY']*mem_all.layer.loop_dim_size['FX'])>1:
            buff_mem=2*mem_all.layer.loop_dim_size['C']*mem_all.layer.loop_dim_size['FY']*mem_all.layer.loop_dim_size['FX']
        elif mem_all.layer.layer_attrs['operator_type']=='depthwise_conv_2d':
            buff_mem=mem_all.layer.loop_dim_size['FY']*(mem_all.layer.loop_dim_size['OY']+sum(mem_all.layer.padding['IY']))+mem_all.layer.loop_dim_size['FY']
        # buff for each core
        num_cores=mem_all.layer.layer_attrs['num_cores'] if 'num_cores' in mem_all.layer.layer_attrs else 8
        buff_mem*=num_cores
        # bias
        if 'bias_add' in mem_all.layer.layer_attrs and mem_all.layer.layer_attrs['attrs']['bias_add']:
            buff_mem+=mem_all.layer.loop_dim_size['K']*4
        if 'batchnorm' in mem_all.layer.layer_attrs and mem_all.layer.layer_attrs['attrs']['batchnorm']:
            buff_mem+=mem_all.layer.loop_dim_size['K']*4*2
        db_support=True
    return buff_mem,db_support


def cost_model():
    return SpecializedLatencyCostModel(
        [
            #default_ideal_temporal_cycles,
            calc_multiplicity_l2_and_transfer_overheads,
            calc_computational_cycles,
            calc_loading_and_latency1,
            calc_offloading_and_latency2,
        ],
        buffer_size_and_db_mem_func
    )
