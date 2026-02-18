import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import Tensor, nn
from typing import (
    Any,
    List, 
    Union, 
    Optional, 
    Callable, 
    Sequence, 
    Literal
)
from dataclasses import dataclass
from jaxtyping import Bool, Float, Int
import einops

THETA_INIT = 0.1
device="cuda"
# https://github.com/timaeus-research/devinterp/blob/main/src/devinterp/utils.py#L279
def cycle(iterable, limit=None):
    """
    Use this function to cycle through a dataloader. Unlike itertools.cycle, this function doesn't cache
    values in memory.

    Note: Be careful with cycling a shuffled interable. The shuffling will be different for each loop dependent on the seed
    state, unlike with itertools.cycle.

    :param iterable: Iterable to cycle through
    :param limit: Number of cycles to go through. If None, cycles indefinitely.
    """
    index = 0
    if limit is None:
        limit = float("inf")
    while True:
        for x in iterable:
            if index >= limit:
                return
            else:
                yield x
            index += 1

def linear_lr(step, steps):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0

def rectangle(x: Tensor, width: float = 1.0) -> Tensor:
    """
    Returns the rectangle function value, i.e. K(x) = 1[|x| < width/2], as a float.
    """
    return (x.abs() < width / 2).float()


class Heaviside(torch.autograd.Function):
    """
    Implementation of the Heaviside step function, using straight through estimators for the derivative.

        forward:
            H(z,θ,ε) = 1[z > θ]

        backward:
            dH/dz := None
            dH/dθ := -1/ε * K(z/ε)

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: Tensor, theta: Tensor, eps: float) -> Tensor:
        # Save any necessary information for backward pass
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        # Compute the output
        return (z > theta).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:
        # Retrieve saved tensors & values
        (z, theta) = ctx.saved_tensors
        eps = ctx.eps
        # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
        grad_z = 0.0 * grad_output
        grad_theta = -(1.0 / eps) * rectangle((z - theta) / eps) * grad_output
        grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary

        return grad_z, grad_theta_agg, None

class JumpReLU(torch.autograd.Function):
    """
    Implementation of the JumpReLU function, using straight through estimators for the derivative.

        forward:
            J(z,θ,ε) = z * 1[z > θ]

        backward:
            dJ/dθ := -θ/ε * K((z - θ)/ε)
            dJ/dz := 1[z > θ]

            where K is the rectangle kernel function with width 1, centered at 0: K(u) = 1[|u| < 1/2]
    """

    @staticmethod
    def forward(ctx: Any, z: Tensor, theta: Tensor, eps: float) -> Tensor:
        # Save any necessary information for backward pass
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        # Compute the output
        return z * (z > theta).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:
        # Retrieve saved tensors & values
        (z, theta) = ctx.saved_tensors
        eps = ctx.eps
        # Compute gradient of the loss with respect to z (no STE) and theta (using STE)
        grad_z = (z > theta).float() * grad_output
        grad_theta = -(theta / eps) * rectangle((z - theta) / eps) * grad_output
        grad_theta_agg = grad_theta.sum(dim=0)  # note, sum over batch dim isn't strictly necessary
        return grad_z, grad_theta_agg, None

@dataclass
class ToySAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    sparsity_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    ste_epsilon: float = 0.01

class ToySAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    # def __init__(self, cfg: ToySAEConfig, model: ToyModel) -> None:
    def __init__(self, cfg: ToySAEConfig) -> None:
        super(ToySAE, self).__init__()
        # self.model = model.requires_grad_(False)
        # self.model.W.data[1:] = self.model.W.data[0]
        # self.model.b_final.data[1:] = self.model.b_final.data[0]
        self.n_inst = cfg.n_inst
        self.cfg = cfg


        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty((cfg.n_inst, cfg.d_in, cfg.d_sae))
        ))
        self._W_dec = None if self.cfg.tied_weights else nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_sae, cfg.d_in))))

        self.b_enc = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty((cfg.n_inst, cfg.d_sae))
        ))
        self.b_dec = nn.Parameter(nn.init.kaiming_uniform_(
            torch.empty((cfg.n_inst, cfg.d_in))
        ))

        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """
        Returns decoder weights, normalized over the autoencoder input dimension.
        """
        # You'll fill this in later
        # return self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps)
        return self.W_dec / (torch.square(self.W_dec).sum(dim=-1, keepdim=True).sqrt() + self.cfg.weight_normalize_eps)


    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, "batch inst"],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict:       dict of different loss terms, each having shape (batch_size, n_inst)
            loss:            total loss (i.e. sum over terms of loss dict), same shape as loss terms
            acts_post:       autoencoder latent activations, after applying ReLU
            h_reconstructed: reconstructed autoencoder input
        """
        # You'll fill this in later

        acts_post = F.relu(einops.einsum(self.W_enc, (h - self.b_dec), "inst d_in d_sae, batch inst d_in -> batch inst d_sae") + self.b_enc)
        h_reconstructed = einops.einsum(self.W_dec_normalized, acts_post, "inst d_sae d_in, batch inst d_sae -> batch inst d_in") + self.b_dec

        l2_loss = torch.square(h-h_reconstructed).mean(dim=-1)
        l1_loss = torch.abs(acts_post).sum(dim=-1)

        return {"L_reconstruction":l2_loss, "L_sparsity":l1_loss}, (l2_loss + self.cfg.sparsity_coeff * l1_loss), acts_post, h_reconstructed


    def optimize(
        self,
        generate_batch: Callable,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float]=constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
        hidden_sample_size: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons
            hidden_sample_size: size of hidden value sample we add to the logs (for visualization)

        Returns:
            data_log:           dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # betas=(0.0, 0.999)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists of dicts to store data we'll eventually be plotting
        data_log = []

        for step in progress_bar:
            # Resample dead latents
            if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                frac_active_in_window = torch.stack(frac_active_list[-resample_window:], dim=0)
                if resample_method == "simple":
                    self.resample_simple(frac_active_in_window, resample_scale, generate_batch)
                elif resample_method == "advanced":
                    self.resample_advanced(frac_active_in_window, resample_scale, generate_batch)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model
            with torch.inference_mode():
                h = generate_batch()

            # Optimize
            loss_dict, loss, acts, _ = self.forward(h)
            torch.cuda.empty_cache()
            loss.mean(0).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them directly (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized.data

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and log a bunch of values for creating plots / animations
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=step_lr,
                    loss=loss.mean(0).sum().item(),
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                with torch.inference_mode():
                    loss_dict, loss, acts, h_r = self.forward(
                        h := generate_batch()
                    )
                    torch.cuda.empty_cache()
                    
                data_log.append(
                    {
                        "steps": step,
                        "frac_active": (acts.abs() > 1e-8).float().mean(0).detach().cpu(),
                        "loss": loss.detach().cpu(),
                        "h": h.detach().cpu(),
                        "h_r": h_r.detach().cpu(),
                        **{name: param.detach().cpu() for name, param in self.named_parameters()},
                        **{name: loss_term.detach().cpu() for name, loss_term in loss_dict.items()},
                    }
                )
            
            torch.cuda.empty_cache()
        return data_log

    @torch.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vecs
            - Set new values of W_dec and W_enc to be these normalized vecs, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """

        dead_lats = (frac_active_in_window < 1e-8).all(dim=0) # inst d_sae
        n_dead = int(dead_lats.int().sum().item())

        v = torch.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        v_normed = v / (v.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps)

        self.W_dec.data[dead_lats] = v_normed
        self.W_enc.data.transpose(-1,-2)[dead_lats] = (v_normed * resample_scale)
        self.b_enc.data[dead_lats] = 0

    @torch.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples latents that have been dead for `dead_feature_window` steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vecs `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec & W_enc to be these centered & normalized vecs, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        raise NotImplementedError()


class JumpReLUToySAE(ToySAE):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]
    log_theta: Float[Tensor, "inst d_sae"]
    def __init__(self, cfg: ToySAEConfig, pre_set_bias=None):
        super(ToySAE, self).__init__()
        # self.model = model.requires_grad_(False)
        # self.model.W.data[1:] = self.model.W.data[0]
        # self.model.b_final.data[1:] = self.model.b_final.data[0]

        self.n_inst = cfg.n_inst
        self.cfg = cfg

        self._W_dec = (
            None
            if self.cfg.tied_weights
            else nn.Parameter(nn.init.kaiming_uniform_(torch.empty((cfg.n_inst, cfg.d_sae, cfg.d_in))))
        )
        
        if pre_set_bias is not None:
            self.b_dec = torch.zeros(cfg.n_inst, cfg.d_in).to(device)
            self.b_enc = torch.zeros(cfg.n_inst, cfg.d_sae).to(device)
        else:
            self.b_dec = nn.Parameter(torch.zeros(cfg.n_inst, cfg.d_in))
            self.b_enc = nn.Parameter(torch.zeros(cfg.n_inst, cfg.d_sae))

        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty((cfg.n_inst, cfg.d_in, cfg.d_sae)))
        )
        self.log_theta = nn.Parameter(torch.full((cfg.n_inst, cfg.d_sae), torch.log(torch.tensor(THETA_INIT))))

        self.to(device)

    @property
    def theta(self) -> Float[Tensor, "inst d_sae"]:
        return self.log_theta.exp()

    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, ""],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Same as previous forward function, but allows for gated case as well (in which case we have different
        functional form, as well as a new term "L_aux" in the loss dict).
        """
        h_cent = h - self.b_dec
        
        acts_pre = (
            einops.einsum(
                h_cent, self.W_enc, "batch inst d_in, inst d_in d_sae -> batch inst d_sae"
            )
            + self.b_enc
        )
        # print(self.theta.mean(), self.theta.std(), self.theta.min(), self.theta.max())
        acts_relu = F.relu(acts_pre)
        acts_post = JumpReLU.apply(acts_relu, self.theta, self.cfg.ste_epsilon)

        h_reconstructed = (
            einops.einsum(
                acts_post, self.W_dec, "batch inst d_sae, inst d_sae d_in -> batch inst d_in"
            )
            + self.b_dec
        )

        loss_dict = {
            "L_reconstruction": (h_reconstructed - h).pow(2).mean(-1),
            "L_sparsity": Heaviside.apply(acts_relu, self.theta, self.cfg.ste_epsilon).sum(-1),
        }

        loss = loss_dict["L_reconstruction"] + self.cfg.sparsity_coeff * loss_dict["L_sparsity"]

        return loss_dict, loss, acts_post, h_reconstructed

    @torch.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        replacement_values = torch.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

        # New names for weights & biases to resample
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = (
            resample_scale * replacement_values_normed
        )
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0
        self.log_theta.data[dead_latents_mask] = torch.log(torch.tensor(THETA_INIT))

    @torch.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        generate_batch: Callable
    ) -> None:
        h = generate_batch()
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.cfg.n_inst):
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = torch.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue

            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue

            distn = Categorical(probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            replacement_values = (h - self.b_dec)[replacement_indices, instance]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
            )

            W_enc_norm_alive_mean = (
                self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item()
                if (~is_dead).any()
                else 1.0
            )

            # New names for weights & biases to resample
            self.b_enc.data[instance, dead_latents] = 0.0
            self.log_theta.data[instance, dead_latents] = torch.log(torch.tensor(THETA_INIT))
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_latents] = (
                replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
            )