"""Basic test for FlexModule with random data to verify mechanics work."""

import torch
import torch_geometric
from flexModel.flex_gnn import FlexGNN_GCNConv_GGConv
from flexModel.flex_module import FlexModule


def test_flex_module_forward_pass():
    """Test FlexModule forward pass with random graph structure and embeddings."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    n_genes = 50
    n_reactions = 30
    n_compounds = 20
    n_modules = 10
    batch_size = 8
    gene_edim = 16
    reaction_edim = 32
    
    # Create random graph structure (gene-to-reaction edges)
    # Each gene connects to 1-3 reactions
    eid_g2r_list = []
    for gene_idx in range(n_genes):
        n_edges = torch.randint(1, 4, (1,)).item()
        reaction_indices = torch.randint(0, n_reactions, (n_edges,))
        for rxn_idx in reaction_indices:
            eid_g2r_list.append([gene_idx, rxn_idx])
    eid_g2r = torch.tensor(eid_g2r_list, dtype=torch.long).t()
    
    # Create random reaction-to-reaction edges (sparse connectivity)
    # Each reaction connects to 2-4 other reactions
    eid_r2r_list = []
    for rxn_idx in range(n_reactions):
        n_edges = torch.randint(2, 5, (1,)).item()
        target_reactions = torch.randint(0, n_reactions, (n_edges,))
        for target_idx in target_reactions:
            if target_idx != rxn_idx:  # no self-loops
                eid_r2r_list.append([rxn_idx, target_idx])
    eid_r2r = torch.tensor(eid_r2r_list, dtype=torch.long).t()
    
    # Create random stoichiometry matrix (sparse, with positive and negative entries)
    Mcr = torch.randn(n_compounds, n_reactions) * 0.3
    Mcr[torch.rand(n_compounds, n_reactions) > 0.3] = 0  # Make it sparse
    
    # Create random gene-to-module mapping (binary matrix)
    Mmg = torch.zeros(n_modules, n_genes)
    for module_idx in range(n_modules):
        # Each module has 3-8 genes
        n_genes_in_module = torch.randint(3, 9, (1,)).item()
        gene_indices = torch.randperm(n_genes)[:n_genes_in_module]
        Mmg[module_idx, gene_indices] = 1.0
    
    # Create random module-to-reaction mapping (binary matrix)
    Mmr = torch.zeros(n_modules, n_reactions)
    for module_idx in range(n_modules):
        # Each module has 2-5 reactions
        n_rxns_in_module = torch.randint(2, 6, (1,)).item()
        rxn_indices = torch.randperm(n_reactions)[:n_rxns_in_module]
        Mmr[module_idx, rxn_indices] = 1.0
    
    # Create random module weights for correlation loss
    cor_wts = torch.rand(n_modules) + 0.5  # weights between 0.5 and 1.5
    cor_wts = cor_wts / cor_wts.sum()  # normalize to sum to 1
    
    # Create random gene and reaction embeddings
    gen_emb = torch.randn(n_genes, gene_edim)
    rea_emb = torch.randn(n_reactions, reaction_edim)
    
    # Initialize GNN model
    gnn = FlexGNN_GCNConv_GGConv(
        nr=n_reactions,
        re_edim=reaction_edim,
        ge_edim=gene_edim,
        nlayers=2
    )
    
    # Initialize FlexModule
    model = FlexModule(
        gnn=gnn,
        eid_g2r=eid_g2r,
        eid_r2r=eid_r2r,
        Mcr=Mcr,
        Mmg=Mmg,
        Mmr=Mmr,
        cor_wts=cor_wts,
        gen_emb=gen_emb,
        rea_emb=rea_emb,
        flx_project=False,  # Test without nullspace projection first
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.0,
        lopt_lr=1e-3,
    )
    
    # Create random gene expression data
    ge = torch.randn(batch_size, n_genes).abs()  # positive expression values
    
    # Test forward pass
    print("Testing forward pass...")
    flxs, flxs_p = model.forward(ge)
    
    # Verify output shapes
    assert flxs.shape == (batch_size, n_reactions), f"Expected shape {(batch_size, n_reactions)}, got {flxs.shape}"
    assert flxs_p is None, "Expected flxs_p to be None when flx_project=False"
    print(f"✓ Forward pass successful. Flux shape: {flxs.shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    losses = model.losses(ge, flxs, flxs_p)
    
    # Verify loss shapes
    assert losses.shape == (batch_size, 5), f"Expected loss shape {(batch_size, 5)}, got {losses.shape}"
    print(f"✓ Loss computation successful. Loss shape: {losses.shape}")
    
    # Print loss component means
    loss_names = ["L_fb", "L_pos", "L_cor", "L_sco", "L_ent"]
    print("\nMean loss components:")
    for i, name in enumerate(loss_names):
        print(f"  {name}: {losses[:, i].mean().item():.4f}")
    
    # Test training step
    print("\nTesting training step...")
    batch = (ge,)  # Tuple format expected by training_step
    loss = model.training_step(batch, batch_idx=0)
    
    # Verify loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    print(f"✓ Training step successful. Total loss: {loss.item():.4f}")
    
    # Test with nullspace projection
    print("\n" + "="*60)
    print("Testing with nullspace projection...")
    
    # Create random nullspace projector (simplified - not true nullspace)
    n_null = n_reactions - 5  # Assume nullspace has dimension n_reactions - 5
    NSP = torch.randn(n_null, n_reactions)
    NSP = torch.nn.functional.normalize(NSP, p=2, dim=1)  # Orthonormalize rows
    
    # Reinitialize model with projection
    model_proj = FlexModule(
        gnn=gnn,
        eid_g2r=eid_g2r,
        eid_r2r=eid_r2r,
        Mcr=Mcr,
        Mmg=Mmg,
        Mmr=Mmr,
        cor_wts=cor_wts,
        gen_emb=gen_emb,
        rea_emb=rea_emb,
        flx_project=True,
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.1,
        lopt_lr=1e-3,
        NSP=NSP,
    )
    
    # Test forward pass with projection
    flxs_proj, flxs_p_proj = model_proj.forward(ge)
    
    # Verify output shapes
    assert flxs_proj.shape == (batch_size, n_reactions), f"Expected shape {(batch_size, n_reactions)}, got {flxs_proj.shape}"
    assert flxs_p_proj.shape == (batch_size, n_reactions), f"Expected projected flux shape {(batch_size, n_reactions)}, got {flxs_p_proj.shape}"
    print(f"✓ Forward pass with projection successful")
    print(f"  Flux shape: {flxs_proj.shape}")
    print(f"  Projected flux shape: {flxs_p_proj.shape}")
    
    # Test loss computation with projection
    losses_proj = model_proj.losses(ge, flxs_proj, flxs_p_proj)
    assert losses_proj.shape == (batch_size, 5)
    print(f"✓ Loss computation with projection successful")
    
    print("\nMean loss components (with projection):")
    for i, name in enumerate(loss_names):
        print(f"  {name}: {losses_proj[:, i].mean().item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_flex_module_forward_pass()
