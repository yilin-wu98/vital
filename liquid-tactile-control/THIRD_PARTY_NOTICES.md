# Third-party source

The experiment drivers and checkpoint adapter derive from the local
CMU-IntentLab/deoxys_control checkout. The original Apache 2.0 license is
retained in `LICENSE`.

`patches/deoxys-continuous-gripper.patch` is the existing local Deoxys gripper
extension, captured against that repository's base commit.

`liquid_tactile_control/policy_backend` contains the required portions of the
local CMU-IntentLab/diffusion4robotics checkout: the policy agent, diffusion
U-Net with tactile guidance, ResNet features, model base class, and image
transforms. Its MIT license is retained in that directory. Source comments
retain the Sudeep Dasari and Diffusion Policy attribution.

The bundled policy backend was copied from base commit
`4f705a3b1d8963c2bd018785bd13b06cadde2312` including the existing local
modifications to `agent.py` and `models/diffusion_unet.py`. Publishing only that
commit would omit the guidance implementation.

The Robometer client is a small protocol adapter. Robometer model/server
source and model weights are not redistributed in this package.

Publication cleanup changes include package imports, deferred hardware
imports, explicit machine configuration, shared AnyTouch checkpoint loading,
checkpoint target remapping, removal of redundant pretrained-weight
downloads, and extraction of the shared tactile-context functions.
The experiment control loops, policy guidance math, and checkpoint tensor
names are retained.
