"""Marketing a novel the legitimate way: assets the author posts themselves, with disclosure.

This package generates promotional material and a rules-respecting posting plan. It deliberately has
no ability to create accounts, post to third-party platforms, or message strangers: see policy.py for
why, and for the boundary encoded in code.
"""
from .channels import CHANNELS, channel, channels_for
from .launchkit import LaunchKit, build_launch_kit, posting_plan
from .policy import ALLOWED, DISCLOSURE, REFUSED, check_action

__all__ = ["CHANNELS", "channel", "channels_for", "LaunchKit", "build_launch_kit", "posting_plan",
           "ALLOWED", "REFUSED", "DISCLOSURE", "check_action"]
