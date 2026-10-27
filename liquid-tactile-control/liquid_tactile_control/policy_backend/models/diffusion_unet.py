# Publication cleanup (2026-10-27): adapted from diffusion4robotics/data4robotics/models/diffusion_unet.py.
# U-Net implementation from: Diffusion Policy Codebase (Chi et al; arXiv:2303.04137)

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
import torch
import numpy as np
import torch.nn as nn
from liquid_tactile_control.policy_backend.agent import Agent

from typing import Union, Optional
import torch
import torch.nn as nn
import math

from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
            in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
            The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of diffusion parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        global_cond: Optional[torch.Tensor] = None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif isinstance(timesteps, torch.Tensor) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        assert isinstance(timesteps, torch.Tensor)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature: torch.Tensor = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class DiffusionUnetAgent(Agent):
    def __init__(self, features, shared_mlp, odim, n_cams, use_obs,
                 ac_dim, ac_chunk, train_diffusion_steps, eval_diffusion_steps,
                 imgs_per_cam=1, dropout=0, share_cam_features=False,
                 feat_batch_norm=True, noise_net_kwargs=dict()):
        super().__init__(features, None, shared_mlp, odim, n_cams,
                         use_obs, imgs_per_cam, dropout, share_cam_features,
                         feat_batch_norm)
        self.noise_net = ConditionalUnet1D(input_dim=ac_dim,
                                           global_cond_dim=self.obs_enc_dim,
                                           **noise_net_kwargs)

        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

        assert eval_diffusion_steps <= train_diffusion_steps, "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
                                    num_train_timesteps=train_diffusion_steps,
                                    beta_start=0.0001,
                                    beta_end=0.02,
                                    beta_schedule="squaredcos_cap_v2",
                                    clip_sample=True,
                                    set_alpha_to_one=True,
                                    steps_offset=0,
                                    prediction_type="epsilon"
                                    )

    def forward(self, imgs, obs, ac_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        timesteps = torch.randint(low=0, high=self._train_diffusion_steps, size=(B,),
                                  device=device).long()

        # diffusion unet logic assumes [B, T, adim]
        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        noise = torch.randn_like(actions)

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        noise_pred = self.noise_net(noise_acs, timesteps, s_t)

        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum((1,2))    # mask the loss to only consider "real" acs
        return loss.mean()

    def get_actions(self, imgs, obs, n_steps=None):
        # get observation encoding and sample noise
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        noise_actions = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert n_steps <= self._train_diffusion_steps, \
                  f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = self.diffusion_schedule.alphas_cumprod.to(device)
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net(noise_actions, batched_timestep, s_t)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(model_output=noise_pred,
                                                timestep=timestep,
                                                sample=noise_actions).prev_sample

        # return final action post diffusion
        return noise_actions

    def get_actions_with_guidance(
        self,
        imgs,
        obs,
        reward_fn,
        guidance_scale=1.5,
        ss=4,
        n_steps=None,
        guidance_every=1,
        guidance_min_alpha=0.0,
        grad_clip=None,
        lpb_scale=False,
        guidance_start_step=0,
        normalize_grad=False,
    ):
        """
        DDIM denoising with classifier-based guidance following the DynaGuide formulation
        (Du & Song, 2025 — https://arxiv.org/abs/2506.13922).

        At each denoising timestep t:
          1. Predict noise ε = noise_net(x_t, t, s_t).
          2. Reconstruct x̂₀ = (x_t − √(1−ᾱₜ)·ε) / √ᾱₜ  [clamped to (−1, 1)].
          3. Compute reward gradient  g = ∇_{x̂₀} r(x̂₀)  via autograd.grad
             (faster than .backward() — only computes the one gradient we need;
              allow_unused=True gracefully handles any broken grad path).
          4. Modify the noise prediction:
               ε_guided = ε − scale · (√(1−ᾱ)/√ᾱ) · g
             (stop-gradient approximation: treats ε as constant w.r.t. x_t).
          5. Take one DDIM step with ε_guided.
          6. Repeat steps 1–5 `ss` times at the same timestep t (stochastic sampling).

        WHY REWARD CAN GO DOWN
        ─────────────────────
        The effective x̂₀ step size is  Δx̂₀ = (1−ᾱ)/ᾱ · scale · ‖∇r‖.
        At early (high-noise) denoising steps ᾱ → 0, so:
          • (1−ᾱ)/ᾱ → ∞  (e.g. ~14,000× at the first DDIM step with squaredcos schedule)
          • x̂₀ reconstruction is unreliable (amplified noise), so ∇r points randomly
          • Combined: guidance at high noise is random and extremely large → reward drops

        TWO FIXES
        ─────────
        1. ``guidance_min_alpha`` — skip guidance when ᾱ < threshold.
           Keeps only the reliable, low-noise part of the denoising process.
           Suggested value: 0.1 (skips first ~half of steps with squaredcos schedule).

        2. ``grad_clip`` — clip ‖∇_{x̂₀} r‖ to at most this value before scaling.
           Prevents a single large gradient from dominating.
           Suggested value: 1.0.

        LATENCY OPTIMIZATIONS
        ─────────────────────
        ``guidance_every`` — apply WM guidance at every this many DDIM timesteps
        (default 1 = every step). Non-guided steps take one unguided DDIM step (ss
        not repeated), halving WM calls with guidance_every=2.

        Expected savings vs baseline (T=16, ss=4, guidance_every=1, guidance_min_alpha=0):
          guidance_every=2                   → −41 % wall time
          guidance_min_alpha=0.1             → −50 % WM calls (skip first ~8 steps)
          guidance_every=2 + min_alpha=0.1   → −70 % WM calls

        Args:
            imgs: camera images dict, same format as get_actions.
            obs: (B, obs_dim) proprio state tensor.
            reward_fn: callable(x0_pred) -> scalar tensor with grad.
                x0_pred is (B, ac_chunk, ac_dim) in policy normalized [−1, 1] space.
                Must be differentiable. If grad path is broken, guidance is silently
                skipped for that step (allow_unused=True).
            guidance_scale: multiplier on the gradient (DynaGuide default: 1.5).
            ss: stochastic-sampling repetitions per guided timestep (DynaGuide default: 4).
            n_steps: override eval diffusion steps (≤ train steps).
            guidance_every: apply WM guidance at every this many DDIM timesteps (default 1).
                Non-guided steps take one unguided DDIM step. guidance_every=2 halves WM
                calls; skipped steps still advance the denoising trajectory normally.
            guidance_min_alpha: skip guidance when ᾱₜ < this value (default 0.0 = no skip).
                Set to 0.1 to skip the noisiest ~half of steps where x̂₀ is unreliable
                and the effective gradient step is astronomically large.
            grad_clip: if not None, clip ‖∇_{x̂₀} r‖₂ to this max norm before applying
                guidance.  Suggested: 1.0.  Prevents single large gradients from
                overshooting and crashing reward.
            lpb_scale: if True, use LPB-style scale √(1−ᾱₜ) instead of the default
                DynaGuide-style √(1−ᾱₜ)/√ᾱₜ.  The LPB scale is bounded in [0,1] at
                all noise levels, so it never explodes at early steps — guidance_min_alpha
                is unnecessary when this is enabled.  The effective gradient magnitude is
                smaller by √ᾱₜ, so guidance_scale may need to be increased (e.g. 5–10×).
                Reference: LPB (Shi et al.) — diffusion_unet_hybrid_image_policy.py.
            guidance_start_step: only apply guidance when step_idx >= this value (default
                0 = guide from the first denoising step).  step_idx counts from 0
                (highest noise / t=T) to eval_steps-1 (lowest noise / t≈0), so setting
                guidance_start_step > 0 skips the noisiest early steps — equivalent to
                LPB's guidance_start_timestep.  Complementary to guidance_min_alpha; both
                can be set simultaneously.
            normalize_grad: if True, normalize ∇_{x̂₀} r to unit norm before scaling
                (FGSM / signed-gradient style).  This decouples guidance_scale from the
                reward gradient magnitude: guidance_scale then directly sets the step size
                in x̂₀ space regardless of how small ‖∇r‖ is.  Useful when the cosine-
                similarity gradient is tiny (e.g., 1e-4) and guidance has no visible
                effect.  With normalize_grad=True, guidance_scale ≈ 0.05–0.2 is a
                reasonable starting range (5–20% of the [−1,1] action range per step).
                Note: grad_clip is applied first (if set), then normalization — so they
                are redundant; prefer normalize_grad alone.
        Returns:
            (B, ac_chunk, ac_dim) denoised action tensor in normalized space.
        """
        B, device = obs.shape[0], obs.device
        _sync_pre = device.type == "cuda"
        if _sync_pre:
            torch.cuda.synchronize()
        _t0_obs = time.perf_counter()
        with torch.no_grad():
            s_t = self._shared_forward(imgs, obs)
        if _sync_pre:
            torch.cuda.synchronize()
        _t_obs_enc = time.perf_counter() - _t0_obs

        noise_actions = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)

        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert n_steps <= self._train_diffusion_steps, \
                f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        self.diffusion_schedule.set_timesteps(eval_steps)
        # Move scheduler's internal alphas_cumprod to device so that diffusion_schedule.step()
        # does not cause a CPU/CUDA device mismatch (mirrors the same line in get_actions).
        self.diffusion_schedule.alphas_cumprod = self.diffusion_schedule.alphas_cumprod.to(device)
        alphas_cumprod = self.diffusion_schedule.alphas_cumprod

        # Tracking lists for visualization (mirrors DynaGuide's corrections/guidance/diffusion lists)
        # Each list has one entry per guided (timestep, ss_iter) pair.
        self.corrections_list = []   # x̂₀ predicted clean action at each guided iter
        self.guidance_list = []      # applied guidance term: scale_factor·scale·grad
        self.diffusion_list = []     # diffusion direction: √(1−ᾱₜ)·ε (unguided)
        self.rewards_list = []       # scalar reward at each guided iter

        # Per-component latency accumulators (seconds)
        _t = {
            "noise_net":  0.0,
            "x0_recon":   0.0,
            "reward_fwd": 0.0,
            "backward":   0.0,
            "ddim_step":  0.0,
        }
        _sync = device.type == "cuda"

        def _cuda_sync():
            if _sync:
                torch.cuda.synchronize()

        _t0_total = time.perf_counter()

        for step_idx, timestep in enumerate(self.diffusion_schedule.timesteps):
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            alpha_prod_t = alphas_cumprod[timestep]
            sqrt_one_minus_alpha = (1.0 - alpha_prod_t).sqrt()

            # Opt: sparse guidance — skip WM at non-guided DDIM steps OR when ᾱ is too
            # small (high-noise regime where x̂₀ is unreliable and step size explodes).
            alpha_ok = (alpha_prod_t.item() >= guidance_min_alpha)
            step_ok  = (step_idx % guidance_every == 0)
            start_ok = (step_idx >= guidance_start_step)
            apply_guidance = alpha_ok and step_ok and start_ok
            inner_iters = ss if apply_guidance else 1

            for _ in range(inner_iters):
                # ── stop-gradient through the noise_net ─────────────────────────
                _cuda_sync(); _t1 = time.perf_counter()
                with torch.no_grad():
                    noise_pred = self.noise_net(noise_actions, batched_timestep, s_t)
                _cuda_sync(); _t2 = time.perf_counter()

                if apply_guidance:
                    with torch.no_grad():
                        x0_raw = (noise_actions - sqrt_one_minus_alpha * noise_pred) / alpha_prod_t.sqrt()
                        x0_raw = x0_raw.clamp(-1.0, 1.0)
                    _cuda_sync(); _t3 = time.perf_counter()

                    x0_pred = x0_raw.detach().requires_grad_(True)
                    reward = reward_fn(x0_pred)
                    _cuda_sync(); _t4 = time.perf_counter()

                    # Use autograd.grad (faster: no .grad accumulation on WM params).
                    # allow_unused=True: silently skip if reward has no path to x0_pred
                    # (e.g. when WM params carry requires_grad but x0_pred path is broken).
                    grad_x0 = None
                    if reward is not None:
                        (grad_x0,) = torch.autograd.grad(
                            reward.sum(), x0_pred, allow_unused=True
                        )
                    _cuda_sync(); _t5 = time.perf_counter()

                    if grad_x0 is not None:
                        if grad_clip is not None:
                            # Clip per-sample gradient norm to prevent overshoot
                            g_norm = grad_x0.norm()
                            if g_norm > grad_clip:
                                grad_x0 = grad_x0 * (grad_clip / g_norm.clamp(min=1e-8))
                        if normalize_grad:
                            # Normalize to unit norm (FGSM / signed-gradient style):
                            # guidance_scale becomes a direct step size in x̂₀ space,
                            # independent of ‖∇r‖.  Use small guidance_scale (0.05–0.2).
                            grad_x0 = grad_x0 / (grad_x0.norm() + 1e-8)
                        # LPB scale: √(1−ᾱ) — bounded, stable at all noise levels.
                        # Default (DynaGuide) scale: √(1−ᾱ)/√ᾱ — explodes at high noise.
                        if lpb_scale:
                            scale_factor = sqrt_one_minus_alpha
                        else:
                            scale_factor = sqrt_one_minus_alpha / alpha_prod_t.sqrt()
                        applied_guidance = scale_factor * guidance_scale * grad_x0
                        noise_pred_guided = noise_pred - applied_guidance
                        self.rewards_list.append(
                            reward.detach().mean().item() if reward is not None else float("nan")
                        )
                        self.guidance_list.append(applied_guidance.detach().clone())
                    else:
                        noise_pred_guided = noise_pred
                        applied_guidance = torch.zeros_like(noise_pred)
                        self.rewards_list.append(float("nan"))
                        self.guidance_list.append(applied_guidance.detach().clone())

                    self.corrections_list.append(x0_raw.detach().clone())
                    self.diffusion_list.append((sqrt_one_minus_alpha * noise_pred).clone())
                    _t["x0_recon"]   += _t3 - _t2
                    _t["reward_fwd"] += _t4 - _t3
                    _t["backward"]   += _t5 - _t4
                else:
                    noise_pred_guided = noise_pred

                # DDIM step with (optionally guided) noise prediction
                _cuda_sync(); _t6 = time.perf_counter()
                with torch.no_grad():
                    noise_actions = self.diffusion_schedule.step(
                        model_output=noise_pred_guided,
                        timestep=timestep,
                        sample=noise_actions,
                    ).prev_sample
                _cuda_sync(); _t7 = time.perf_counter()

                _t["noise_net"] += _t2 - _t1
                _t["ddim_step"] += _t7 - _t6

        total = time.perf_counter() - _t0_total
        measured = sum(_t.values()) + _t_obs_enc
        def _is_guided(i):
            a_ok = alphas_cumprod[self.diffusion_schedule.timesteps[i]].item() >= guidance_min_alpha
            return (i % guidance_every == 0) and a_ok and (i >= guidance_start_step)

        n_guided = sum(ss if _is_guided(i) else 0 for i in range(eval_steps))
        n_total_iters = sum(ss if _is_guided(i) else 1 for i in range(eval_steps))

        scale_tag = f"lpb(√(1−ᾱ))" if lpb_scale else "dynaguide(√(1−ᾱ)/√ᾱ)"
        print(
            f"\n[guidance latency]  total={total*1e3:.1f}ms  "
            f"guided_iters={n_guided}  total_iters={n_total_iters}  "
            f"(T={eval_steps} × ss={ss}, every={guidance_every}, "
            f"min_α={guidance_min_alpha}, start_step={guidance_start_step}, scale={scale_tag})"
        )
        col_w = [14, 12, 15, 15]
        headers = ["component", "total (ms)", "per-iter (ms)", "% of measured"]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
        print("  " + header_line)
        print("  " + "-" * len(header_line))

        pct_obs = _t_obs_enc / measured * 100 if measured > 0 else 0.0
        print(f"  {'obs_enc':<{col_w[0]}}  {_t_obs_enc*1e3:>{col_w[1]}.2f}  "
              f"{'(one-off)':>{col_w[2]}}  {pct_obs:>{col_w[3]}.1f}%")

        for name, v in _t.items():
            tot_ms = v * 1e3
            denom = n_guided if name in ("reward_fwd", "backward", "x0_recon") else n_total_iters
            per_ms = (v / denom * 1e3) if denom > 0 else 0.0
            pct = v / measured * 100 if measured > 0 else 0.0
            print(f"  {name:<{col_w[0]}}  {tot_ms:>{col_w[1]}.2f}  "
                  f"{per_ms:>{col_w[2]}.2f}  {pct:>{col_w[3]}.1f}%")

        overhead = total - measured
        pct_oh = overhead / total * 100 if total > 0 else 0.0
        n_denom = max(n_total_iters, 1)
        print(f"  {'overhead':<{col_w[0]}}  {overhead*1e3:>{col_w[1]}.2f}  "
              f"{overhead/n_denom*1e3:>{col_w[2]}.2f}  {pct_oh:>{col_w[3]}.1f}%")
        print()

        return noise_actions

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
