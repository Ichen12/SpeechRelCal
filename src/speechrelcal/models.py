"""The three learned composers used in Table 1."""
import math
import torch

class PositiveLogLinear(torch.nn.Module):
    def __init__(self,factors):
        super().__init__();self.raw_weight=torch.nn.Parameter(torch.full((factors,),math.log(math.expm1(1))))
    def forward(self,p,m):return (torch.nn.functional.softplus(self.raw_weight)*p.clamp_min(1e-7).log()*m).sum(-1)

class DeepSets(torch.nn.Module):
    def __init__(self, factors: int, element_dimension: int, hidden: int):
        super().__init__()
        self.factors = factors
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(factors + 2, element_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(element_dimension, element_dimension),
            torch.nn.ReLU(),
        )
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(element_dimension, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            self.factors, device=probabilities.device, dtype=probabilities.dtype
        ).expand(*probabilities.shape[:-1], self.factors, self.factors)
        elements = torch.cat(
            [probabilities.unsqueeze(-1), mask.unsqueeze(-1), identity], dim=-1
        )
        pooled = (self.phi(elements) * mask.unsqueeze(-1)).sum(dim=-2)
        return self.rho(pooled).squeeze(-1)

class RelationAwareDeepSets(torch.nn.Module):
    def __init__(self, factors, width):
        super().__init__()
        self.factors = factors
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(factors + 4, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1))

    def forward(self, inputs, mask):
        identity = torch.eye(self.factors, device=inputs.device, dtype=inputs.dtype)
        identity = identity.expand(*mask.shape[:-1], self.factors, self.factors)
        elements = torch.cat([inputs[..., :1], mask.unsqueeze(-1), identity, inputs[..., 1:]], -1)
        return self.rho((self.phi(elements) * mask.unsqueeze(-1)).sum(-2)).squeeze(-1)


def make_model(name, factors, width=16):
    if name == 'log_linear':
        return PositiveLogLinear(factors)
    if name == 'deepsets':
        return DeepSets(factors, width, width)
    if name == 'deepsets_r':
        return RelationAwareDeepSets(factors, width)
    raise ValueError(name)
