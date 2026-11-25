# Todo

- move model graph definitions to python side
- make sure models don't hardcode anything. e.g. embeddinggemma2.py _VisionGraph._rotate computes cos/sin values in python and passes them as constants instead of constructing them within the graph
- add GitHub CI (build + C++ tests + planner oracle)
- clean up tracked root-level clutter (move example runners/scripts out of root)
