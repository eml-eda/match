from match.codegen.temporal_mapping_engine.zigzag_temporal_mapping_engine import ZigZagEngine
from match.codegen.temporal_mapping_engine.temporal_mapping_engine import TemporalMappingEngine
    
TEMPORAL_MAPPING_ENGINE={
    "zigzag":ZigZagEngine,
}

def get_temporal_mapping_engine(engine_name:str=""):
    if engine_name not in TEMPORAL_MAPPING_ENGINE:
        return TemporalMappingEngine
    else:
        assert issubclass(TEMPORAL_MAPPING_ENGINE[engine_name],TemporalMappingEngine)
        return TEMPORAL_MAPPING_ENGINE[engine_name]