import torch
class BiLinAntisymmetricFunc(torch.nn.Module):
    def __init__(self, d, k=8, rank=64):
        super(BiLinAntisymmetricFunc, self).__init__()
        self.g = torch.nn.Sequential(
            torch.nn.Linear(d, k),
            torch.nn.ReLU(),
            torch.nn.Linear(k, k),
            torch.nn.ReLU(),
            torch.nn.Linear(k, 1),
        )
        self.U = torch.nn.Parameter(torch.randn(k, d, rank))
        self.V = torch.nn.Parameter(torch.randn(k, d, rank))
        self.alpha = torch.nn.Parameter(torch.ones(k))
    
    def forward(self, x1, x2):
        """Forward pass computing antisymmetric bilinear form.
        
        Args:
            x1: Tensor of shape B x nR x d
            x2: Tensor of shape B x nR x d
            
        Returns:
            Tensor of shape B x nR
        """
        # Build antisymmetric matrices M_k = U_k V_kᵀ − V_k U_kᵀ
        M = torch.matmul(self.U, self.V.transpose(-1, -2)) \
            - torch.matmul(self.V, self.U.transpose(-1, -2))  # K×d×d
        g_out1 = self.g(x1).squeeze(-1)  # B x nR
        g_out2 = self.g(x2).squeeze(-1)  # B x nR
        # Vectorised bilinear: einsum replaces Python loop over K matrices
        # x1 @ M -> B×nR×K×d, element-wise * x2 -> sum over d -> B×nR×K
        bili = torch.einsum('bnd,kdf,bnf,k->bn', x1, M, x2, self.alpha)
        return g_out1 - g_out2 + bili  # B x nR
