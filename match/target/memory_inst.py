
from typing import Callable, Dict, List, NamedTuple, Tuple

class PortConnection:
    def __init__(self,rw_port_number:int=1,r_port_number:int=0,w_port_number:int=0):
        self.rw_port_number=rw_port_number
        self.r_port_number=r_port_number
        self.w_port_number=w_port_number
        self.reading_port=None
        self.writing_port=None

    def define_ports(self,r_ports:int=0,w_ports:int=0,rw_ports:int=1):
        if self.r_port_number>r_ports:
            self.r_port_number=r_ports
        if self.w_port_number>w_ports:
            self.w_port_number=w_ports
        if self.rw_port_number>rw_ports:
            self.rw_port_number=rw_ports
        self.reading_port={
            "type":"r",
            "number":self.r_port_number,
        } if self.r_port_number>0 else {
            "type":"rw",
            "number":self.rw_port_number,
        }
        self.writing_port={
            "type":"w",
            "number":self.w_port_number,
        } if self.w_port_number>0 else {
            "type":"rw",
            "number":self.rw_port_number,
        }
        assert ((bool(self.r_port_number) + bool(self.w_port_number)) == 2) or bool(self.rw_port_number)

def no_buffer(*args):
    return 0

class MemoryInst:
    def __init__(self,name:str="l1_mem",k_bytes:int=1,r_bw:int=32,w_bw:int=32,r_ports:int=0,w_ports:int=0,
                 rw_ports:int=1,operands:List[str]=[],double_buffering_support:bool=False,
                 buffer_for_layer_func:Callable=no_buffer,used_ports:Dict[str,Tuple[PortConnection]]={}):
        self.name=name
        self.k_bytes=k_bytes
        self.r_bw=r_bw
        self.w_bw=w_bw
        self.r_ports=r_ports
        self.w_ports=w_ports
        self.rw_ports=rw_ports
        self.operands=operands
        self.double_buffering_support=double_buffering_support
        self.buffer_for_layer_func=buffer_for_layer_func
        self.used_ports=dict()
        assert ((bool(self.r_ports) + bool(self.w_ports)) == 2) or bool(self.rw_ports)
        for port_op in [op for op in self.operands if op not in self.used_ports]:
            self.used_ports[port_op]=(
                PortConnection(rw_port_number=1 if rw_ports>0 else 0,r_port_number=1 if r_ports>0 else 0,w_port_number=1 if w_ports>0 else 0),
                PortConnection(rw_port_number=1 if rw_ports>0 else 0,r_port_number=1 if r_ports>0 else 0,w_port_number=1 if w_ports>0 else 0),
            )
        for port_op in self.operands:
            self.used_ports[port_op][0].define_ports(r_ports=r_ports,w_ports=w_ports,rw_ports=rw_ports)
            if port_op=="O":
                self.used_ports[port_op][1].define_ports(r_ports=r_ports,w_ports=w_ports,rw_ports=rw_ports)