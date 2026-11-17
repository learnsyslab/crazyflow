from hypothesis import Phase, settings

# The explain phase reruns failing examples about a thousand times, which takes minutes for known
# deviations. TODO: Enable all phases once the expected failures are fixed
settings.register_profile("fuzzing", phases=[phase for phase in Phase if phase != Phase.explain])
settings.load_profile("fuzzing")
