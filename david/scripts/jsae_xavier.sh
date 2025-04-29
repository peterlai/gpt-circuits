# Layer 0 2e-4, 4e-4, 6e-4, 8e-4
# Layer 1 3e-3
# Layer 2 3e-3
# Layer 3 2e-2, 4e-2, 6e-2, 8e-2

# Layer 0 = 2e-4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=2e-4,3e-3,3e-3,2e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=2e-4,3e-3,3e-3,4e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=2e-4,3e-3,3e-3,6e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=2e-4,3e-3,3e-3,8e-2

# Layer 0 = 4e-4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=4e-4,3e-3,3e-3,2e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=4e-4,3e-3,3e-3,4e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=4e-4,3e-3,3e-3,6e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=4e-4,3e-3,3e-3,8e-2

# Layer 0 = 6e-4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=6e-4,3e-3,3e-3,2e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=6e-4,3e-3,3e-3,4e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=6e-4,3e-3,3e-3,6e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=6e-4,3e-3,3e-3,8e-2

# Layer 0 = 8e-4
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=8e-4,3e-3,3e-3,2e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=8e-4,3e-3,3e-3,4e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=8e-4,3e-3,3e-3,6e-2
/workspace/HOME/guest/.conda/envs/spar/bin/python -m training.sae.jsae_concurrent --max_steps=20_000 --sparsity=8e-4,3e-3,3e-3,8e-2
