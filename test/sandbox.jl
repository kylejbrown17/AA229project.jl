using Pkg
Pkg.activate("/home/kylebrown/Repos/AA229_project")
using POMDPs, QuickPOMDPs, POMDPSimulators, QMDP
using SARSOP
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
        n_obstacles_x=1,
        n_obstacles_y=1,
        obs_width = [2;2],
        obs_offset = [1;1],
        env_pad = [0;0],
        )
    factory_env

    N = 2
    M = 1
    E = typeof(factory_env)
    pomdp = FactoryPOMDP{E,N,M}(
        starts = [1,2],
        goals = [50,51],
        # intruder_start_dist = [vertices(factory_env)],
        intruder_start_dist = [[1],],
        deadline=4,
        env = factory_env,
        action_cache = factory_env.edge_cache,
        obs_map = generate_obs_model(factory_env)
    )

    # m = DiscreteExplicitPOMDP(S,A,O,T,Z,R,γ)
    @requirements_info QMDPSolver() pomdp
    @requirements_info SARSOPSolver() pomdp
    # solver = SARSOPSolver()
    solver = QMDPSolver()
    policy = solve(solver, pomdp)

    n_states(pomdp)
    states(pomdp)
    for (i,s) in enumerate(states(pomdp))
        @assert i == stateindex(pomdp,s)
    end
    for (i,a) in enumerate(actions(pomdp))
        @assert i == actionindex(pomdp,a)
    end
    for (i,o) in enumerate(observations(pomdp))
        @assert i == obsindex(pomdp,o)
    end

    s0_dist = initialstate_distribution(pomdp)
    rng = MersenneTwister(0)
    s0 = rand(rng,s0_dist)
    rand(rng,s0_dist)
    support(s0_dist)
    pdf(s0_dist,s0_dist.vals[1])
    mode(s0_dist)
    # mean(s0_dist)

    a = ((0,0),(0,0))
    actionindex(pomdp,a)

    sp_dist = transition(pomdp,s,a)
    sp = rand(rng,sp_dist)
    rand(rng,sp_dist)
    support(sp_dist)
    pdf(sp_dist,sp_dist.vals[1])
    mode(sp_dist)
    # mean(sp_dist)

    o = Observation(state=sp)
    o.intruders_observed
    obs_dist = observation(pomdp,a,sp)
    rand(rng,obs_dist)
    support(obs_dist)
    pdf(obs_dist,obs_dist.val)
    mode(obs_dist)
    mean(obs_dist)

    possible_actions = actions(pomdp,s)
    actions(pomdp)
    generate_action_index_dict(pomdp)

    reward(pomdp,s,a)
end
