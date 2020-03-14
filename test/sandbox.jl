using Pkg
Pkg.activate("/home/kylebrown/Repos/AA229_project")
using POMDPs, QuickPOMDPs, POMDPSimulators, QMDP
using AA229project
using LightGraphs
using LinearAlgebra
using GraphUtils
using TaskGraphs
using Parameters
using Distributions
using Random

# Tiger POMDP example
let
    S = [:left, :right]
    A = [:left, :right, :listen]
    O = [:left, :right]
    γ = 0.95

    function T(s, a, sp)
        if a == :listen
            return s == sp
        else # a door is opened
            return 0.5 #reset
        end
    end

    function Z(a, sp, o)
        if a == :listen
            if o == sp
                return 0.85
            else
                return 0.15
            end
        else
            return 0.5
        end
    end

    function R(s, a)
        if a == :listen
            return -1.0
        elseif s == a # the tiger was found
            return -100.0
        else # the tiger was escaped
            return 10.0
        end
    end

    m = DiscreteExplicitPOMDP(S,A,O,T,Z,R,γ)

    solver = QMDPSolver()
    policy = solve(solver, m)

    rsum = 0.0
    for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
        println("s: $s, b: $([pdf(b,s) for s in S]), a: $a, o: $o")
        rsum += r
    end
    println("Undiscounted reward was $rsum.")
end

# Factory environment
let
    factory_env = construct_regular_factory_world(;
        n_obstacles_x=3,
        n_obstacles_y=3,
        obs_width = [2;2],
        obs_offset = [1;1],
        env_pad = [0;0],
        env_scale = 0.5,
        transition_time=2.0
        )
    factory_env

    pomdp = FactoryPOMDP(
        starts = [1,2],
        goals = [50,51],
        n_intruders = 1,
        env = factory_env,
        action_cache = factory_env.edge_cache,
        obs_map = generate_obs_model(factory_env)
    )

    n_states(pomdp)
    states(pomdp)

    s = FactoryState(
        (RobotState((4,1),0),RobotState((1,2),0),),
        (RobotState((2,1),0),)
    )
    stateindex(pomdp,s)

    a = ((0,0),(0,0))
    actionindex(pomdp,a)

    sp = transition(pomdp,s,a)
    sp.robot_states,sp.intruder_states

    o = Observation(state=sp)
    o.intruders_observed
    obs_dist = observation(pomdp,a,sp)

    possible_actions = actions(pomdp,s)

    s0_dist = initialstate_distribution(pomdp)
    rng = MersenneTwister(0)
    s0 = rand(rng,s0_dist)

    reward(pomdp,s,a)

end
