"""Test that the model can train and fit random data."""

import pytest
import torch
import torch.nn.functional as F
from flexModel import FlexModule
from flexModel.flex_gnn import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW, FlexGNN_Disc_GGConv, FlexGNN_Disc_GGConv_LW


def create_flex_module(n_genes=50, n_reactions=30, gene_edim=16, reaction_edim=32, use_layer_weights=False, use_disc=False):
    """Create a FlexModule with random graph structure and embeddings."""
    n_compounds = n_reactions // 2
    n_modules = max(5, n_genes // 10)
    
    # Create random gene-to-reaction edges
    eid_g2r_list = []
    for gene_idx in range(n_genes):
        n_edges = torch.randint(1, 4, (1,)).item()
        reaction_indices = torch.randint(0, n_reactions, (n_edges,))
        for rxn_idx in reaction_indices:
            eid_g2r_list.append([gene_idx, rxn_idx])
    eid_g2r = torch.tensor(eid_g2r_list, dtype=torch.long).t()
    
    # Create random reaction-to-reaction edges
    eid_r2r_list = []
    for rxn_idx in range(n_reactions):
        n_edges = torch.randint(2, 5, (1,)).item()
        target_reactions = torch.randint(0, n_reactions, (n_edges,))
        for target_idx in target_reactions:
            if target_idx != rxn_idx:
                eid_r2r_list.append([rxn_idx, target_idx])
    eid_r2r = torch.tensor(eid_r2r_list, dtype=torch.long).t()
    
    # Create random matrices
    Mcr = torch.randn(n_compounds, n_reactions) * 0.3
    Mcr[torch.rand(n_compounds, n_reactions) > 0.3] = 0
    
    Mmg = torch.zeros(n_modules, n_genes)
    for module_idx in range(n_modules):
        n_genes_in_module = torch.randint(3, 9, (1,)).item()
        gene_indices = torch.randperm(n_genes)[:n_genes_in_module]
        Mmg[module_idx, gene_indices] = 1.0
    
    Mmr = torch.zeros(n_modules, n_reactions)
    for module_idx in range(n_modules):
        n_rxns_in_module = torch.randint(2, 6, (1,)).item()
        rxn_indices = torch.randperm(n_reactions)[:n_rxns_in_module]
        Mmr[module_idx, rxn_indices] = 1.0
    
    cor_wts = torch.ones(n_modules)
    
    gen_emb = torch.randn(n_genes, gene_edim)
    rea_emb = torch.randn(n_reactions, reaction_edim)
    
    # Initialize GNN (with or without layer weights)
    if use_disc:
        f_disc = torch.rand(eid_r2r.shape[1])
        if use_layer_weights:
            gnn = FlexGNN_Disc_GGConv_LW(
                nr=n_reactions, f_disc_orig=f_disc, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2
            )
        else:
            gnn = FlexGNN_Disc_GGConv(
                nr=n_reactions, f_disc_orig=f_disc, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2
            )
    else:
        if use_layer_weights:
            gnn = FlexGNN_GCNConv_GGConv_LW(
                nr=n_reactions, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2
            )
        else:
            gnn = FlexGNN_GCNConv_GGConv(
                nr=n_reactions, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2
            )
    
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
        flx_project=False,
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.0,
        lopt_lr=1e-3,
    )
    
    return model, n_genes, n_reactions


@pytest.mark.parametrize('use_disc', [False, True])
def test_model_overfits_random_data(use_disc):
    """Test that model can overfit a small random training set (standard FlexGNN).
    
    If the model cannot reduce loss on random data, there's likely
    a bug in the training setup, loss computation, or backpropagation.
    """
    print("\n🧪 Testing standard FlexGNN training on random data...")
    
    # Create model with realistic dimensions
    torch.manual_seed(42)
    model, n_genes, n_reactions = create_flex_module(n_genes=64, n_reactions=128, use_layer_weights=False, use_disc=use_disc)
    
    # Create fixed random gene expression data
    n_samples = 16
    ge = torch.randn(n_samples, n_genes).abs()
    
    # Get initial loss
    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()
    
    print(f"Initial loss: {initial_loss_value:.4f}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    n_steps = 50
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        loss = model.training_step((ge,), step)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")
    
    final_loss = losses[-1]
    print(f"\nFinal loss: {final_loss:.4f}")
    print(f"Loss reduction: {initial_loss_value - final_loss:.4f} ({100*(1 - final_loss/initial_loss_value):.1f}% decrease)")
    print(f"Loss trajectory: [{losses[0]:.3f} → {losses[9]:.3f} → {losses[19]:.3f} → {losses[29]:.3f} → {losses[39]:.3f} → {losses[49]:.3f}]")
    
    # Assertions
    assert final_loss < initial_loss_value, \
        f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"
    
    # Should achieve some reduction (at least 10% to verify training works)
    reduction_ratio = final_loss / initial_loss_value
    assert reduction_ratio < 0.9, \
        f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease. Expected at least 10%."
    
    print("✅ Model successfully overfits random data - training mechanics work!")


@pytest.mark.parametrize('use_disc', [False, True])
def test_model_with_layer_weights_overfits(use_disc):
    """Test that model with layer weights can overfit random data (FlexGNN_LW).
    
    Verifies that the layer-weighted variant also trains correctly.
    """
    print("\n🧪 Testing FlexGNN with layer weights on random data...")
    
    # Create model with layer weights and realistic dimensions
    torch.manual_seed(42)
    model, n_genes, n_reactions = create_flex_module(n_genes=64, n_reactions=128, use_layer_weights=True, use_disc=use_disc)
    
    # Create fixed random gene expression data
    n_samples = 16
    ge = torch.randn(n_samples, n_genes).abs()
    
    # Get initial loss
    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()
    
    print(f"Initial loss: {initial_loss_value:.4f}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    n_steps = 50
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        loss = model.training_step((ge,), step)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")
    
    final_loss = losses[-1]
    print(f"\nFinal loss: {final_loss:.4f}")
    print(f"Loss reduction: {initial_loss_value - final_loss:.4f} ({100*(1 - final_loss/initial_loss_value):.1f}% decrease)")
    print(f"Loss trajectory: [{losses[0]:.3f} → {losses[9]:.3f} → {losses[19]:.3f} → {losses[29]:.3f} → {losses[39]:.3f} → {losses[49]:.3f}]")
    
    # Assertions
    assert final_loss < initial_loss_value, \
        f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"
    
    # Should achieve some reduction (at least 10% to verify training works)
    reduction_ratio = final_loss / initial_loss_value
    assert reduction_ratio < 0.9, \
        f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease. Expected at least 10%."
    
    print("✅ FlexGNN with layer weights successfully overfits random data!")


if __name__ == "__main__":
    test_model_overfits_random_data()
    test_model_with_layer_weights_overfits()
