import torch
print("torch version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch built with CUDA:", torch.version.cuda)
print("device count:", torch.cuda.device_count())

import ray
ray.init()
print(ray.cluster_resources())
print(ray.available_resources())


ray.shutdown()
ray.init(num_gpus=1)
print("Now I set num_gpus = 1")
print(ray.cluster_resources())
print(ray.available_resources())

print("Shutting down ray.")
ray.shutdown()

print("Ray has been shutdown. The real work starts NOW")