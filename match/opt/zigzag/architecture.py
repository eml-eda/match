
from math import floor
from typing import Any, List
from match.target.memory_inst import MemoryInst, PortConnection
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.opt.temporal.loma.engine import NoValidLoopOrderingFoundException

class ZigZagSoC:

    def __init__(self, soc_name: str="MATCH"):
        self.soc_name = soc_name

    def zigzag_port_def(self, port:PortConnection, read_def:bool=True):
        if port.reading_port is None:
            return None
        elif read_def:
            return f"{port.reading_port['type']}_port_{port.reading_port['number']}"
        return f"{port.writing_port['type']}_port_{port.writing_port['number']}"
    
    def get_zigzag_op_from_tensor_type(self, t_type: str="out"):
        if t_type=="output":
            return "O"
        elif t_type=="var":
            return "I1"
        elif t_type=="const":
            return "I2"

    def rename_from_tensor_type_to_zigzag_operands(self, tensor_types: List[str]=[]):
        return ["O" if t_type=="output" else "I1" if t_type=="var" else "I2" for t_type in tensor_types if t_type!="intermediate"]
    
    def get_memory_hierarchy(self, multiplier_array, platform_memories: List[MemoryInst]=[]):
        """Memory hierarchy variables"""
        """ size=#bit, bw=#bit"""
        memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
        for plat_mem in platform_memories:
            mem_inst = MemoryInstance(
                name = plat_mem.name,
                size = floor(( plat_mem.k_bytes * 1024)* 8),
                r_bw=floor(plat_mem.r_bw),
                w_bw=floor(plat_mem.w_bw),
                r_cost=100,
                w_cost=110,
                area=0,
                r_port=floor(plat_mem.r_ports),
                w_port=floor(plat_mem.w_ports),
                rw_port=floor(plat_mem.rw_ports),
                latency=1, # TODO: Unused in ZigZag, should be > 1
                double_buffering_support=plat_mem.double_buffering_support
            )
            port_alloc=tuple([
                {
                    "fh":self.zigzag_port_def(port=plat_mem.used_ports[t_type][0],read_def=False),
                    "tl":self.zigzag_port_def(port=plat_mem.used_ports[t_type][0],read_def=True),
                    "fl":self.zigzag_port_def(port=plat_mem.used_ports[t_type][1],read_def=False),
                    "th":self.zigzag_port_def(port=plat_mem.used_ports[t_type][1],read_def=True),
                }
                for t_type in plat_mem.tensor_types
            ])
            memory_hierarchy_graph.add_memory(
                memory_instance=mem_inst,
                operands=self.rename_from_tensor_type_to_zigzag_operands(plat_mem.tensor_types),
                port_alloc=port_alloc,
                served_dimensions="all",
            )

        return memory_hierarchy_graph


    # Non usati per il momento
    def get_operational_array(self, optimal_spatial_mapping: List[Any]=[]):
        """Multiplier array variables"""
        multiplier_input_precision = [8, 8]
        multiplier_energy = 0.04
        multiplier_area = 1
        dimensions={f"D{idx+1}":opt_spat[1] for idx,opt_spat in enumerate(optimal_spatial_mapping)}
        multiplier = Multiplier(
            multiplier_input_precision, multiplier_energy, multiplier_area
        )
        multiplier_array = MultiplierArray(multiplier, dimensions)

        return multiplier_array


    # Sovrascritto
    def get_dataflows(self, optimal_spatial_mapping: List[Any]=[]):
        return [{f"D{idx+1}":(opt_spat[0],opt_spat[1]) for idx,opt_spat in enumerate(optimal_spatial_mapping)}]
    
    def get_core(self, id,
                    optimal_spatial_mapping: List[Any]=[],
                    platform_memories: List[MemoryInst]=[]):
        operational_array = self.get_operational_array(optimal_spatial_mapping=optimal_spatial_mapping)
        #get the memory hierarchy, from the l2 to the register level
        memory_hierarchy = self.get_memory_hierarchy(operational_array, platform_memories=platform_memories)
        dataflows = self.get_dataflows(optimal_spatial_mapping=optimal_spatial_mapping)
        core = Core(id, operational_array, memory_hierarchy, dataflows)
        return core

    def get_accelerator(self, optimal_spatial_mapping: List[Any]=[],
                        platform_memories: List[MemoryInst]=[]):
        """Generate a ZigZag architecture"""
        cores = {self.get_core(1,optimal_spatial_mapping=optimal_spatial_mapping,platform_memories=platform_memories)}
        acc_name = 'MATCH'
        return Accelerator(acc_name, cores)
