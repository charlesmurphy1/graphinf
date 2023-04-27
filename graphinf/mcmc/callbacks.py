# from _graphinf import mcmc as _mcmc

# CallBack = _mcmc.callbacks.CallBack
# CollectLikelihoodOnSweep = _mcmc.callbacks.CollectLikelihoodOnSweep
# CollectPriorOnSweep = _mcmc.callbacks.CollectPriorOnSweep
# CollectJointOnSweep = _mcmc.callbacks.CollectJointOnSweep
# CheckConsistencyOnSweep = _mcmc.callbacks.CheckConsistencyOnSweep
# CheckSafetyOnSweep = _mcmc.callbacks.CheckSafetyOnSweep
# VerboseDisplay = _mcmc.callbacks.VerboseDisplay
# cb = _mcmc.callbacks

# __all__ = (
#     "CallBack",
#     "CollectEdgesOnSweep",
#     "CollectGraphOnSweep",
#     "CollectPartitionOnSweep",
#     "CollectLikelihoodOnSweep",
#     "CollectPriorOnSweep",
#     "CollectJointOnSweep",
#     "CheckConsistencyOnSweep",
#     "CheckSafetyOnSweep",
#     "VerboseDisplay",
# )


# def CollectEdgesOnSweep(labeled=False, nested=False):
#     if nested and labeled:
#         return cb._CollectNestedBlockLabeledEdgeMultiplicityOnSweep()
#     elif not nested and labeled:
#         return cb._CollectBlockLabeledEdgeMultiplicityOnSweep()
#     return cb._CollectEdgeMultiplicityOnSweep()


# def CollectGraphOnSweep(labeled=False, nested=False):
#     if nested and labeled:
#         return cb._CollectNestedBlockLabeledGraphOnSweep()
#     elif not nested and labeled:
#         return cb._CollectBlockLabeledGraphOnSweep()
#     return cb._CollectGraphOnSweep()


# def CollectPartitionOnSweep(nested=False, type="community"):
#     if nested and type == "community":
#         return cb._CollectNestedPartitionOnSweepForCommunity()
#     elif nested and type == "reconstruction":
#         return cb._CollectNestedPartitionOnSweepForReconstruction()
#     elif not nested and type == "community":
#         return cb._CollectPartitionOnSweepForCommunity()
#     return cb._CollectPartitionOnSweepForReconstruction()
