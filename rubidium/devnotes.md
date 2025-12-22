### Notes
- graphing.py is currently purely a path handler, however the video path handling is splatter between scanning and "graphing" and video still.
- i dont think it makes any sense to use the stream from crowsnest, the delay is high (which shouldnt matter) however the "jitter" in the actual delay seems to be too high for proper usage.
- print/scan mode currently dont support named sections, the only reason is that i havent tested it, it should just work tbh
- latency mode isnt meant as something final, optimally it would be a small routine integrated into scan mode, perhaps we can also use openCV to detect the actual start/end frames by choosing the right spacing from the "brim" 
- configview was introduced later and print/scan mode still need to switch to using that instead (base.py)


### TODO
- exclude object building: right winding and closed polygon → *(slicers always output them closed, mirror that)*
- unify path handling between scan and video