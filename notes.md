FIXES TO ADD:
[x] - inline and ? for loop stuff
[x] - understand constant data issue

ROADMAP:
tag 0: pulp-nn working again[
    [] - add apis again
    [] - constants formatting
    [] - golden check
]
tag 1: cpu working easy version[
    [x] - fix static version to emit loop
    [x] - translate match_node into the basic schedule
    [x] - - add for each op the "translation" as a basic schedule
    [x] - - start with a basic schedule with multiple blocks
    [x] - add concept of instructions
    [x] - emit c code for a simple basic schedule
]
tag 2: cpu working fusing ops[
    [] - understand if there's a chance to fuse operations
    [] - understand data dependendencies between ops
    [] - theres chance between tied ops fuse blocks into a single one
]
tag 3: parallel cpu[
    [] - parallelization passes
    [] - - add a pass to understand at which level we can parallelize
    [] - - modify the schedule to fit for parallelization
    [] - - understand best parallel # of units
]
tag 4: backend support[
    [] - design how to define the backend function(can be also ISA)
    [] -- constraints(data must be in a certain format or data must be in L1)
    [] -- suggested solution is to define the backend function as a python function
    [] -- understand it with ast, then get the ast of the schedule and if the backend
    [] -- is contained in the schedule theres two things
    [] -- understand if the constraints are matched
    [] -- pass to transform the schedule with the backend function
]
tag 5: neopt-scheduler[
    [] - define a genetic algorithm
    [] - - mutations
    [] - - tiling
    [] - - reordering
    [] - - unrolling
    [] - - lower schedule pass to compute idx, get values etc.
    [] - - lower schedule pass to reorder instr
    [] - backends
    [] - - control if other backend function are matched at the same place
    [] - -- if so open another branch of the schedule search
]
tag 6: llm[
    [] - runtime working?
    [] - can match complex patterns?
    [] - simple basic-schedule cpu working?
]




add -> conv -> relu

add_ch
add_h
add_w

inp_h_idx = out_h_idx*stride + fil_h*dilations
out_h_idx * stride = inp_h_idx - fil_h*dilations
out_h_idx = (inp_h_idx - fil_h*dilations)/stride

for inp_ch_idx in inp_ch:
    for inp_h_idx in inp_h:
        for inp_w_idx in inp_w:
            add_res[inp_ch_idx][inp_h_idx][inp_w_idx]+=add_consts[inp_ch_idx][inp_h_idx][inp_w_idx]


for b in b_size:
    for o_c in o_ch_size:
        for o_h in o_h_size:
            for o_w in o_w_size:
                sum = 0
                ...
                out = sum

for b in b_size:
    for o_h in o_h_size:
        for o_c in o_c_size:
            bias_val = biases[o_c]
            for o_w in o_w_size:
                out += bias_val

