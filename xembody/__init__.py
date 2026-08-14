"""xembody -- portable core.

Nothing in this package imports mujoco, robosuite, LIBERO or torch. Modules take
plain values and return plain values, so they drop into any host simulator or a
real robot. Host-specific code lives in `examples/`.
"""
