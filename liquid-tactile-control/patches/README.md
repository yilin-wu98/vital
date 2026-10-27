# Deoxys continuous gripper compatibility

The liquid drivers construct `FrankaInterface(...,
gripper_mode="continuous_smooth")`. This option comes from the local
modifications to `deoxys/deoxys/franka_interface/franka_interface.py`; it is
not present in base commit
`5ca447a5df7d1cefadf43fbfba7fd5f147383d7c`.

`deoxys-continuous-gripper.patch` preserves that file's complete existing diff,
including the smoothing parameters, command throttling, and gripper-state
reset. It is a compatibility artifact, not a newly tested controller.
The patch was checked against the base file in an isolated directory, and
the resulting file matched the current research source byte for byte.

For a fresh Deoxys checkout at the base commit, inspect the patch and check
whether it applies:

```bash
git apply --check /path/to/liquid-tactile-control/patches/deoxys-continuous-gripper.patch
```

If the checkout needs these changes, apply it there with `git apply`.
The existing research-machine checkout already has them; do not apply the
patch again. Installing this Python package does not patch Deoxys.

No gripper command or robot motion was executed during publication cleanup.
