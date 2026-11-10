"""crochetsim: pattern -> physically simulated crochet video.

The yarn is one chain of nodes.  Nothing in the scene has velocity or stored
elastic energy, so a node moves only while a constraint that reaches it is
being violated, and the only things that violate constraints are the tools
(the hook and the tail gripper).  See README.md.
"""
