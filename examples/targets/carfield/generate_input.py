import numpy as np

shape = (1, 64, 16, 16)
file_path = "./blocks_fp16/resnet/input.txt"
data = np.random.rand(*shape).astype(np.float16)
np.savetxt(file_path, data.flatten(), delimiter='\n', fmt='%f', header=f"Input data ({','.join(map(str, shape))})")