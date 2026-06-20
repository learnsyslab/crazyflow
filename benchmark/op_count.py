import crazyflow  # noqa: F401, ensure gymnasium envs are registered
from crazyflow.sim import Sim


def main():
    """Main entry point for profiling."""
    sim = Sim(n_worlds=1, n_drones=1, dynamics="first_principles", control="attitude")

    compiled_reset = sim._reset.lower(sim.data, sim.default_data, None).compile()
    compiled_step = sim._step.lower(sim.data, 1).compile()
    print(f"Reset cost analysis: {compiled_reset.cost_analysis()}")
    print(f"Step cost analysis: {compiled_step.cost_analysis()}")


if __name__ == "__main__":
    main()
