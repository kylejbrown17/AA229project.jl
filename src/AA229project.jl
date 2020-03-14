module AA229project

using POMDPs
using POMDPModelTools
using GraphUtils
using LightGraphs
using LinearAlgebra
using GraphUtils
using Parameters
using Distributions
using Random


export
    FactoryPOMDP

@with_kw struct FactoryPOMDP{E}
    # rewards
    r_goal::Float64             = 100.0
    r_delay::Float64            = -10.0
    r_too_close::Float64        = -10.0
    r_collision::Float64        = -1000.0
    r_step::Float64             = -1.0
    discount_factor::Float64    = 0.95
    # observations
    true_positive_rate::Float64     = 1.0
    false_positive_rate::Float64     = 0.0
    # env definition
    goals::Vector{Int}      = Int[]
    starts::Vector{Int}     = Int[]
    goal_times::Vector{Int} = map(g->0,goals)
    deadline::Int           = 100 # simulation ends at T=100
    n_robots::Int           = length(goals)
    n_intruders::Int        = 1
    env::E                  = Graph()
    action_cache::Vector{Set{Tuple{Int,Int}}} = Vector{Set{Tuple{Int,Int}}}()
    # Observation
    obs_map::Vector{Vector{Int}} = Vector{Vector{Int}}()
end
POMDPs.discount(pomdp::FactoryPOMDP) = pomdp.discount_factor

export
    RobotState,
    FactoryState,
    FactoryStateSpace,
    FactoryStateDistribution,
    ActionSpace,
    get_possible_actions

const VTX = Tuple{Int,Int}
# robot state is an int
@with_kw struct RobotState
    vtx::VTX = (-1,-1)
    t::Int      = -1
end
Base.show(io::IO,s::RobotState) = print(io,"RobotState(vtx=$(s.vtx),t=$(s.t))")
@with_kw struct FactoryState{N,M}
    robot_states::NTuple{N,RobotState}    = (RobotState(),)
    intruder_states::NTuple{M,RobotState} = (RobotState(),)
    t::Int                                 = 0
end
Base.show(io::IO,s::FactoryState) = print(io,
    "FactoryState:\n  robots: $(map(i->i.vtx,s.robot_states))",
    "\n  intruders: $(map(i->i.vtx,s.intruder_states))",
    "\n  t=$(s.robot_states[1].t)")

@with_kw struct FactoryStateSpace
    vtxs::Vector{VTX}       = Vector{VTX}()
    n_robots::Int           = 0
    n_intruders::Int        = 0
end
# TODO include t in n_states?
POMDPs.n_states(m::FactoryPOMDP,args...) = nv(m.env)^(m.n_robots+m.n_intruders)
POMDPs.states(m::FactoryPOMDP) = FactoryStateSpace(m.env.vtxs,m.n_robots,m.n_intruders)
function POMDPs.stateindex(m::FactoryPOMDP, state::FactoryState)
    n = length(m.env.vtxs)
    idx = 0
    i = 0
    for s in [state.robot_states..., state.intruder_states...]
        v = m.env.vtx_map[s.vtx...]
        idx += v*n^i
        i += 1
    end

    idx
end
function Base.rand(rng::AbstractRNG, space::FactoryStateSpace)
    t=0
    FactoryState(
        tuple(map(i->RobotState(rand(rng,space.vtxs),t),1:space.n_robots)),
        tuple(map(i->RobotState(rand(rng,space.vtxs),t),1:space.n_intruders))
        )
end

@with_kw struct FactoryStateDistribution{N,M}
    vtxs::Vector{VTX}                       = Vector{VTX}()
    robot_states::NTuple{N,RobotState}      = (RobotState(),)
    intruder_beliefs::NTuple{M,Categorical} = (Categorical(1),)
end
function POMDPs.initialstate_distribution(m::FactoryPOMDP)
    t0 = 0
    N = m.n_robots
    M = m.n_intruders
    
    robot_states=tuple(map(v->RobotState(m.env.vtxs[v],t0),m.starts)...)
    s0_vector = Vector{FactoryState{N,M}}()
    vtxs = m.env.vtxs
    for vtx_list in Base.Iterators.product(map(i->vtxs,1:M)...)
        push!(s0_vector,
            FactoryState(
                robot_states = robot_states,
                intruder_states = tuple(map(vtx->RobotState(vtx,t0),vtx_list)...),
                t = t0
            )
        )
    end
    # s0_vector = Vector{FactoryState{N,M}}(
    #     map(vtx->FactoryState(
    #         robot_states = robot_states,
    #         intruder_states = tuple(map(i->RobotState(vtx,t0),1:M)...),
    #         t = t0
    #         ), vtxs)
    # )
    SparseCat(s0_vector,ones(length(s0_vector))/length(s0_vector))

    # FactoryStateDistribution(
    #     vtxs=m.env.vtxs,
    #     robot_states=tuple(map(v->RobotState(m.env.vtxs[v],0),m.starts)...),
    #     intruder_beliefs=tuple(map(i->Categorical(nv(m.env)),1:m.n_intruders)...)
    # )
end
function Base.rand(rng::AbstractRNG, d::FactoryStateDistribution)
    FactoryState(
        d.robot_states,
        tuple(map(c->RobotState(vtx=d.vtxs[rand(rng,c)]),d.intruder_beliefs)...)
        )
end

@with_kw struct RobotAction
    vtx::VTX = (-1,-1)
end
struct ActionSpace{T<:Tuple}
    actions::T
end
get_possible_actions(m::FactoryPOMDP,s) = sort(collect(m.action_cache[m.env.vtx_map[s.vtx...]]))
POMDPs.actions(m::FactoryPOMDP,s) = ActionSpace(map(state->get_possible_actions(m,state),s.robot_states))
function POMDPs.actionindex(m::FactoryPOMDP,action::NTuple{N,VTX}) where {N}
    @assert m.n_robots == N
    possible_actions = (
        (-1,0),
        (0,0),
        (0,-1),
        (0,1),
        (1,0)
    )
    idx = 0
    i = 0
    for a in action
        for (j,at) in enumerate(possible_actions)
            if a == at
                idx += j*N^i
                i += 1
                break
            end
        end
    end
    return idx
end

export
    robot_transition

function robot_transition(m::FactoryPOMDP,s,a)
    vtxp = [s.vtx...] + [a...]
    RobotState(VTX(vtxp),s.t+1)
end
function POMDPs.transition(m::FactoryPOMDP,s::S,a) where {S}
    robot_states = tuple(map(sa->robot_transition(m,sa...),zip(s.robot_states,a))...)
    sp_vector = Vector{S}()
    for a_list in Base.Iterators.product(map(state->rand(get_possible_actions(m,state)),s.intruder_states)...)
        push!(sp_vector,
            FactoryState(
                robot_states = robot_states,
                intruder_states = tuple(map(sa->robot_transition(m,sa...),zip(s.intruder_states,a_list))...)
            )
        )
    end
    SparseCat(sp_vector,ones(length(sp_vector))/length(sp_vector))
    # intruder_actions = map(state->rand(get_possible_actions(m,state)),s.intruder_states)
    # sp = FactoryState(
    #     robot_states = tuple(map(sa->robot_transition(m,sa...),zip(s.robot_states,a))...),
    #     intruder_states = tuple(map(sa->robot_transition(m,sa...),zip(s.intruder_states,intruder_actions))...)
    # )
    # return a distribution

end

export
    generate_obs_model,
    Observation,
    ObsDistribution

function generate_obs_model(factory_env;
    o_kernel=[
        0  1  1  1  0;
        1  1  1  1  1;
        1  1  1  1  1;
        1  1  1  1  1;
        0  1  1  1  0;
    ],
    o_ctr = [3,3]
    )
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
    obs_map
end

@with_kw struct Observation{S<:FactoryState,T}
    state::S                = FactoryState()
    intruders_observed::T   = tuple(map(i->false,state.intruder_states)...)
end
@with_kw struct ObsDistribution{O}
    obs::O                      = Observation()
    true_positive_rate::Float64 = 1.0
end
function POMDPs.observation(m::FactoryPOMDP,a,sp)
    intruders_observed = zeros(Bool,length(sp.intruder_states))
    for (i,is) in enumerate(sp.intruder_states)
        if ~intruders_observed[i]
            for rs in sp.robot_states
                if is in m.obs_map[m.env.vtx_map[rs.vtx...]]
                    intruders_observed[i] = true
                    break
                end
            end
        end
    end
    o = Observation(sp,tuple(intruders_observed...))
    # ObsDistribution(obs=o,true_positive_rate=m.true_positive_rate)
    Deterministic(o)
end

function POMDPs.reward(m::FactoryPOMDP,s,a)
    r = 0.0
    for (rs,goal,t_goal) in zip(s.robot_states,m.goals,m.goal_times)
        if m.env.vtx_map[rs.vtx...] == goal
            r += m.r_goal
            continue
        elseif rs.t > t_goal
            r += m.r_delay
        else
            r += m.r_step
        end

        for is in s.intruder_states
            if rs.vtx == is.vtx
                r += m.r_collision
            elseif abs(rs.vtx[1]-is.vtx[1])+abs(rs.vtx[2]-is.vtx[2]) <= 1
                r += m.r_too_close
            end
        end
    end
    return r
end


end # module
