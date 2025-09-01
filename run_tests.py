import os

test_cases = [
    (1024, 128),
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (2048, 256),
    (2048, 512),
    (2048, 1024),
    (2048, 2048),
    (4096, 512),
    (4096, 1024),
    (4096, 2048),
    (4096, 4096)
]

for seq_len, embed_dim in test_cases:
    print(f"Running test for seq_len={seq_len}, embed_dim={embed_dim}")
    
    command = f"python test_driver.py --embed_dim {embed_dim} --seq_len {seq_len}"
    
    os.system(command)
