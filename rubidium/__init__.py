"""Rubidium Klipper extra.

Intended config layout:

    [rubidium]                 # base/shared config
    [rubidium print]           # registers RUBIDIUM_PRINT command
    [rubidium scan]            # registers RUBIDIUM_SCAN command
    [rubidium analyse]         # registers RUBIDIUM_ANALYSE command (optional)

    [rubidium pattern]         # default pattern definition
    [rubidium pattern <name>]  # named pattern definitions

    [rubidium print <name>]    # named print preset (does not register commands)
    [rubidium scan <name>]     # named scan preset (does not register commands)
    [rubidium analyse <name>]  # named analyse preset (does not register commands)
"""

from __future__ import annotations

from typing import Any

from .core.registry import RubidiumRegistry


def load_config(config: Any):
    return RubidiumRegistry(config)

def load_config_prefix(config: Any):
    """Load prefixed [rubidium ...] sections."""
    printer = config.get_printer()
    try: 
        printer.load_object(config, "rubidium")
    except:
        raise config.error('a base rubidium section is required.')

    cfg_name = config.get_name()
    parts = cfg_name.split()
    if len(parts) < 2:
        return None

    kind = parts[1].lower()

    if kind == "print":
        if len(parts) == 2:
            from .modes.print_mode import RubidiumPrint
            return RubidiumPrint(config)
        else:
            raise config.error(f'Invalid section [{cfg_name}], print sections may not be named (yet)')

    if kind == "scan":
        if len(parts) == 2:
            from .modes.scan_mode import RubidiumScan
            return RubidiumScan(config)
        else:
            raise config.error(f'Invalid section [{cfg_name}], scan sections may not be named (yet)')
        
    if kind in ("analyse"):
        if len(parts) == 2:
            from .analysis.analyse_mode import RubidiumAnalyse
            return RubidiumAnalyse(config)
        else:
            raise config.error(f'Invalid section [{cfg_name}], analyse sections may not be named (yet)')

    if kind == "pattern":
        from .patterns.patterns import RubidiumPatternObject
        return RubidiumPatternObject.from_config(config)

    raise config.error(f"Unknown rubidium section kind '{kind}' in [{cfg_name}]")
