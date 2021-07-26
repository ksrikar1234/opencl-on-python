import numpy as np
import pyopencl as cl
import time
from pprint import pprint

a_np = np.random.rand(72000000).astype(np.float32)
b_np = np.random.rand(72000000).astype(np.float32)
#x = json_load("rand_arrays.json")
#a_np = x["x"]
#b_np = x["z"]
#pprint(a_np)
#pprint(b_np)
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=b_np)

#pprint(b_g)
prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = 2*(a_g[gid] + b_g[gid]) + 0.5f*b_g[gid] ;
}
""").build()

st = time.time()
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a_np.shape, None, a_g, b_g, res_g)
opencl_time_taken = time.time() - st
res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

#print(" time take on opencl device =   %.6f sec"  %(opencl_time_taken))
#pprint("The Result on OpenCL device is : ")
#pprint(res_np)

a_np = np.random.rand(72000000).astype(np.float32)

b_np = np.random.rand(72000000).astype(np.float32)
# Check on CPU with Numpy:
pprint("CPU")
cpu_st = time.time()
x = (2*a_np + 2.5*b_np)
cpu_time_taken = time.time() - cpu_st
#print(" time take on CPU =   %.6f sec"  %(cpu_time_taken))
#pprint((x))
pprint(np.linalg.norm(res_np - (x)))
assert np.allclose(res_np, x)

print("cpu_time_taken      = %.6f sec"  %(cpu_time_taken))
print("opencl_time_taken   = %.6f sec"  %(opencl_time_taken))
