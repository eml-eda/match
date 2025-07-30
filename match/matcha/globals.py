# Optimization is performed in TVM transform pass
# This variable is used to store the result of the optimization
# It is set by the MatchOptimizer 
# It is accessed by the graph runtime for c code generation

optimization_result = None

def set_optimization_result(result):
    global optimization_result
    optimization_result = result
    
def get_optimization_result():
    global optimization_result
    return optimization_result