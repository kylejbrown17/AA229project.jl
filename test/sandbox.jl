using Pkg
Pkg.activate("/home/kylebrown/Repos/AA229_project")
using POMDPs, QuickPOMDPs, POMDPSimulators, QMDP
using AA229project
using LightGraphs
using LinearAlgebra
using GraphUtils
using TaskGraphs

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
    factory_env.vtx_map

    # Observation model
    o_kernel = [
        0  1  1  1  0;
        1  1  1  1  1;
        1  1  1  1  1;
        1  1  1  1  1;
        0  1  1  1  0;
    ]
    o_ctr = [3,3]
    x_idxs = 1-o_ctr[1]:size(o_kernel,1)-o_ctr[1]
    y_idxs = 1-o_ctr[2]:size(o_kernel,2)-o_ctr[2]
    obs_map = map(v->Int[],vertices(factory_env))
    for v in vertices(factory_env)
        vtx = factory_env.vtxs[v]
        for (dx,dy) in Base.Iterators.product(x_idxs,y_idxs)
            vtx2 = (vtx[1]+dx, vtx[2]+dy)
            v2 = get(factory_env.vtx_map, vtx2, -1)
            if v2 != v && v2 > 0 && o_kernel[dx+o_ctr[1],dy+o_ctr[2]] > 0
                visible = true
                # exclude blocked cells that are around a corner
                for (x,y) in Base.Iterators.product(min(vtx[1],vtx2[1]):max(vtx[1],vtx2[1]),min(vtx[2],vtx2[2]):max(vtx[2],vtx2[2]))
                    if (x,y) != vtx && (x,y) != vtx2 && get(factory_env.vtx_map,(x,y), -1) <= 0
                        p1 = [vtx...]
                        p2 = [vtx2...]
                        p = [x,y]
                        d = ((p2 - p1)/norm(p2-p1))*dot((p-p1),(p2 - p1)/norm(p2 - p1))
                        if norm(p-p1-d) < 0.5
                            # @show v,v2,p-p1,d
                            visible = false
                        end
                    end
                end
                if visible
                    push!(obs_map[v], factory_env.vtx_map[vtx[1]+dx, vtx[2]+dy])
                end
            end
        end
        sort!(obs_map[v])
    end
    factory_env.vtx_map, obs_map

    # Reward function
    function collision_penalty(vr,vh,G)
        if vr == vh
            return -1000.0
        elseif vh in outneighbors(G,vr)
            return -10.0
        else
            return 0.0
        end
    end
    function goal_reward(v,t,goal,t_goal)
        if v == goal
            return 10.0
        elseif t > t_goal
            return -1.0
        else
            return 0.0
        end
    end

end
