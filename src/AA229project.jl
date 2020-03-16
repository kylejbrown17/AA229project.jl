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

const VTX = Tuple{Int,Int}

export
    RobotState,
    FactoryState,
    FactoryStateSpace

@with_kw struct RobotState
    vtx::VTX    = (-1,-1)
    t::Int      = -1
end
Base.show(io::IO,s::RobotState) = print(io,"RobotState(vtx=$(s.vtx),t=$(s.t))")
@with_kw struct FactoryState{N,M}
    robot_states::NTuple{N,RobotState}      = (RobotState(),)
    intruder_states::NTuple{M,RobotState}   = (RobotState(),)
    t::Int                                  = 0
end
Base.show(io::IO,s::FactoryState) = print(io,
    "FactoryState:\n  robots: $(map(i->i.vtx,s.robot_states))",
    "\n  intruders: $(map(i->i.vtx,s.intruder_states))",
    "\n  t=$(s.robot_states[1].t)")

@with_kw struct FactoryStateSpace{N,M}
    vtxs::Vector{VTX}       = Vector{VTX}()
    vtx_map::Matrix{Int}    = zeros(1,1)
    t_max::Int              = 1
end
Base.length(S::FactoryStateSpace{N,M}) where {N,M} = (S.t_max+1)*length(S.vtxs)^(N+M)
POMDPs.dimensions(S::FactoryStateSpace{N,M}) where {N,M} = 1+N+M
function Base.rand(rng::AbstractRNG, S::FactoryStateSpace{N,M}) where {N,M}
    t=rand(rng,0:S.t_max)
    FactoryState(
        tuple(map(i->RobotState(rand(rng,S.vtxs),t),1:N)),
        tuple(map(i->RobotState(rand(rng,S.vtxs),t),1:M)),
        t=t
        )
end

export
    state_to_idx,
    idx_to_state

function state_to_idx(S::FactoryStateSpace{N,M}, state::FactoryState{N,M}) where {N,M}
    n = length(S.vtxs)
    idx = 1
    i = 0
    for s in reverse([state.robot_states..., state.intruder_states...])
        v = S.vtx_map[s.vtx...]
        idx += (1+S.t_max)*(v-1)*n^i
        i += 1
    end
    idx += state.t

    idx
end
function idx_to_state(S::FactoryStateSpace{N,M}, idx::Int) where {N,M}
    n = length(S.vtxs)
    idx,t = divrem(idx,S.t_max+1)
    i_vtxs = RobotState[]
    for j in 1:M
        idx,v = divrem(idx,n)
        push!(i_vtxs,RobotState(S.vtxs[v+1],t))
    end
    r_vtxs = RobotState[]
    for i in 1:N
        idx,v = divrem(idx,n)
        push!(r_vtxs,RobotState(S.vtxs[v+1],t))
    end
    FactoryState(tuple(reverse(r_vtxs)...),tuple(reverse(i_vtxs)...),t)
end
function Base.iterate(S::FactoryStateSpace{N,M}) where {N,M}
    return iterate(S,0)
end
function Base.iterate(S::FactoryStateSpace{N,M}, iter_state::Int) where {N,M}
    if iter_state >= length(S)
        return nothing
    end
    idx_to_state(S,iter_state), iter_state+1
end

export
    ActionSpace

const Action{N} = NTuple{N,VTX}
struct ActionSpace{T<:Tuple}
    actions::T
end

export
    generate_action_index_dict

function generate_action_index_dict(N)
    possible_actions = (
        (-1,0),
        (0,0),
        (0,-1),
        (0,1),
        (1,0)
    )
    actions = collect(Base.Iterators.product(map(i->possible_actions,1:N)...))[:]
    Dict(a=>i for (i,a) in enumerate(actions))
end

export
    Observation

@with_kw struct Observation{N,M}
    state::FactoryState{N,M} = FactoryState()
    intruders_observed::NTuple{M,Bool} = tuple(map(i->false,state.intruder_states)...)
end

export
    ObservationSpace,
    obs_to_idx,
    idx_to_obs

struct ObservationSpace{N,M}
    S::FactoryStateSpace{N,M}
end
Base.length(O::ObservationSpace{N,M}) where {N,M} = (O.S.t_max+1)*(length(O.S.vtxs)^N)*((length(O.S.vtxs)+1)^M) # one extra "invisible" state for each agent
POMDPs.dimensions(O::ObservationSpace{N,M}) where {N,M} = 1+N+M
function Base.rand(rng::AbstractRNG, O::ObservationSpace{N,M}) where {N,M}
    s = rand(rng,O.S)
    intruders_observed = tuple(rand(rng,Bool,M))
    return Observation{N,M}(s,intruders_observed)
end

function obs_to_idx(O::ObservationSpace{N,M}, o::Observation{N,M}) where {N,M}
    S = O.S
    state = o.state
    n = length(S.vtxs)
    idx = 1
    i = 0
    for (s,t) in zip(reverse(state.intruder_states),reverse(o.intruders_observed))
        v = t ? get(S.vtx_map,s.vtx,n+1) : n+1
        # v = get(S.vtx_map,s.vtx,n+1)
        idx += (1+S.t_max)*(v-1)*(n+1)^i
        i += 1
    end
    k = 0
    for s in reverse(state.robot_states)
        v = S.vtx_map[s.vtx...]
        idx += (1+S.t_max)*((n+1)^i)*(v-1)*n^k
        k += 1
    end
    idx += state.t

    idx
end
function idx_to_obs(O::ObservationSpace{N,M}, idx::Int) where {N,M}
    S = O.S
    n = length(S.vtxs)
    idx,t = divrem(idx,S.t_max+1)
    intruders_visible = ones(Bool,M)
    i_vtxs = RobotState[]
    for j in 1:M
        idx,v = divrem(idx,n+1)
        if v == n
            intruders_visible[j] = false
        end
        push!(i_vtxs,RobotState(get(S.vtxs,v+1,(-1,-1)),t))
    end
    r_vtxs = RobotState[]
    for i in 1:N
        idx,v = divrem(idx,n)
        push!(r_vtxs,RobotState(S.vtxs[v+1],t))
    end
    s = FactoryState(tuple(reverse(r_vtxs)...),tuple(reverse(i_vtxs)...),t)
    o = Observation{N,M}(s,tuple(reverse(intruders_visible)...))
end
function Base.iterate(O::ObservationSpace{N,M}) where {N,M}
    return iterate(O,0)
end
function Base.iterate(O::ObservationSpace{N,M}, iter_state::Int) where {N,M}
    if iter_state >= length(O)
        return nothing
    end
    idx_to_obs(O,iter_state), iter_state+1
end

export
    FactoryPOMDP

@with_kw struct FactoryPOMDP{E,N,M} <: POMDP{FactoryState{N,M},Action{N},Observation{N,M}}
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
    n_robots::Int           = length(goals)
    intruder_start_dist::Vector{Vector{Int}} = Vector{Vector{Int}}()
    n_intruders::Int        = length(intruder_start_dist)
    deadline::Int           = 100 # simulation ends at T=100
    env::E                  = Graph()
    action_cache::Vector{Set{VTX}} = Vector{Set{VTX}}()
    action_index_cache::Dict{NTuple{N,VTX},Int} = generate_action_index_dict(n_robots)
    # Observation
    obs_map::Vector{Vector{Int}} = Vector{Vector{Int}}()
end
POMDPs.discount(pomdp::FactoryPOMDP) = pomdp.discount_factor

export
    get_possible_actions

function POMDPs.isterminal(m::FactoryPOMDP,s::FactoryState)
    terminal = true
    if s.t > m.deadline
        return true
    end
    for (rs,goal,t_goal) in zip(s.robot_states,m.goals,m.goal_times)
        if m.env.vtx_map[rs.vtx...] != goal || rs.t < t_goal
            terminal = false # not all robots are done
        end
        for is in s.intruder_states
            if rs.vtx == is.vtx
                return true # collision
            end
        end
    end
    terminal
end

POMDPs.states(m::FactoryPOMDP{E,N,M}) where {E,N,M} = FactoryStateSpace{N,M}(vtxs = m.env.vtxs,vtx_map = m.env.vtx_map, t_max = m.deadline)
POMDPs.n_states(m::FactoryPOMDP) = length(states(m))
function POMDPs.stateindex(m::FactoryPOMDP, state::FactoryState)
    state_to_idx(states(m),state)
    # S = states(m)
    # n = length(S.vtxs)
    # idx = 1
    # i = 0
    # # for s in [state.robot_states..., state.intruder_states...]
    # for s in reverse([state.robot_states..., state.intruder_states...])
    #     v = m.env.vtx_map[s.vtx...]
    #     idx += (1+S.t_max)*(v-1)*n^i
    #     i += 1
    # end
    # idx += state.t
    #
    # idx
end
function POMDPs.initialstate_distribution(m::FactoryPOMDP{E,N,M}) where {E,N,M}
    t0 = 0

    robot_states=tuple(map(v->RobotState(m.env.vtxs[v],t0),m.starts)...)
    s0_vector = Vector{FactoryState{N,M}}()
    # vtxs = m.env.vtxs
    for vtx_list in Base.Iterators.product(map(v_list->map(v->m.env.vtxs[v],v_list), m.intruder_start_dist)...)
    # for vtx_list in Base.Iterators.product(map(i->vtxs,1:M)...)
        push!(s0_vector,
            FactoryState(
                robot_states = robot_states,
                intruder_states = tuple(map(vtx->RobotState(vtx,t0),vtx_list)...),
                t = t0
            )
        )
    end
    SparseCat(s0_vector,ones(length(s0_vector))/length(s0_vector))
end
get_possible_actions(m::FactoryPOMDP,s) = sort(collect(m.action_cache[m.env.vtx_map[s.vtx...]]))
get_possible_actions(m::FactoryPOMDP) = ((-1,0),(0,0),(0,-1),(0,1),(1,0))

function POMDPs.actions(m::FactoryPOMDP,s::FactoryState)
    collect(Base.Iterators.product(
        map(state->get_possible_actions(m,state),s.robot_states)...
        ))[:]
end
function POMDPs.actions(m::FactoryPOMDP{E,N,M}) where {E,N,M}
    collect(Base.Iterators.product(
        map(state->get_possible_actions(m),1:N)...
        ))[:]
end
function POMDPs.actionindex(m::FactoryPOMDP,action::NTuple{N,VTX}) where {N}
    m.action_index_cache[action]
    # @assert m.n_robots == N
    # possible_actions = get_possible_actions(m)
    # idx = 0
    # i = 0
    # for a in action
    #     for (j,at) in enumerate(possible_actions)
    #         if a == at
    #             idx += j*N^i
    #             i += 1
    #             break
    #         end
    #     end
    # end
    # return idx
end

generate_action_index_dict(m::FactoryPOMDP) = generate_action_index_dict(m.n_robots)

export
    robot_transition

function robot_transition(m::FactoryPOMDP,s,a)
    vtxp = VTX([s.vtx...] + [a...])
    if get(m.env.vtx_map,vtxp,-1) <= 0
        vtxp = s.vtx # default to no action
    end
    RobotState(vtxp,s.t+1)
end
function POMDPs.transition(m::FactoryPOMDP,s::S,a) where {S<:FactoryState}
    robot_states = tuple(map(sa->robot_transition(m,sa...),zip(s.robot_states,a))...)
    sp_vector = Vector{S}()
    for a_list in Base.Iterators.product(map(state->get_possible_actions(m,state),s.intruder_states)...)
        push!(sp_vector,
            FactoryState(
                robot_states = robot_states,
                intruder_states = tuple(map(sa->robot_transition(m,sa...),zip(s.intruder_states,a_list))...)
            )
        )
    end
    SparseCat(sp_vector,ones(length(sp_vector))/length(sp_vector))
end


export
    generate_obs_model,
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
    Deterministic(o)
end
POMDPs.observations(m::FactoryPOMDP{E,N,M}) where {E,N,M} = ObservationSpace(states(m))
POMDPs.obsindex(m::FactoryPOMDP{E,N,M},o) where {E,N,M} = obs_to_idx(observations(m),o)


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
