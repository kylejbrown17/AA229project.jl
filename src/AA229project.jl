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
    # n_robots::Int           = 0
    # n_intruders::Int        = 0
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

# struct StateIter
#     s::Int # source state
#     neighbor_list::Vector{Int} # length of target edge list
# end
# struct IterState
#     idx::Int # idx of target node
# end
# ActionIterState() = ActionIterState(0)
function Base.iterate(S::FactoryStateSpace{N,M}) where {N,M}
    return iterate(S,0)
end
function Base.iterate(S::FactoryStateSpace{N,M}, iter_state::Int) where {N,M}
    if iter_state >= length(S)
        return nothing
    end
    idx = iter_state
    idx,t = divrem(idx,S.t_max+1)
    # t,idx = divrem(idx,div(length(S),(S.t_max+1)))
    # @show t
    i_vtxs = RobotState[]
    for j in 1:M
        idx,v = divrem(idx,length(S.vtxs))
        # @show v,j
        push!(i_vtxs,RobotState(S.vtxs[v+1],t))
    end
    r_vtxs = RobotState[]
    for i in 1:N
        idx,v = divrem(idx,length(S.vtxs))
        # @show v,i
        push!(r_vtxs,RobotState(S.vtxs[v+1],t))
    end
    FactoryState(tuple(reverse(r_vtxs)...),tuple(reverse(i_vtxs)...),t), iter_state+1
end

export
    ActionSpace

const Action{N} = NTuple{N,VTX}
struct ActionSpace{T<:Tuple}
    actions::T
end

export
    Observation

@with_kw struct Observation{N,M}
    state::FactoryState{N,M} = FactoryState()
    intruders_observed::NTuple{M,Bool} = tuple(map(i->false,state.intruder_states)...)
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
    deadline::Int           = 100 # simulation ends at T=100
    n_robots::Int           = length(goals)
    n_intruders::Int        = 1
    env::E                  = Graph()
    action_cache::Vector{Set{VTX}} = Vector{Set{VTX}}()
    # action_index_cache::Dict{NTuple{N,VTX},Int}=Dict{NTuple{N,VTX},Int}()
    # Observation
    obs_map::Vector{Vector{Int}} = Vector{Vector{Int}}()
end
POMDPs.discount(pomdp::FactoryPOMDP) = pomdp.discount_factor

export
    get_possible_actions

function POMDPs.isterminal(m::FactoryPOMDP,s::FactoryState)
    terminal = true
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

POMDPs.states(m::FactoryPOMDP{E,N,M}) where {E,N,M} = FactoryStateSpace{N,M}(vtxs = m.env.vtxs,t_max = m.deadline)
POMDPs.n_states(m::FactoryPOMDP) = length(states(m))
function POMDPs.stateindex(m::FactoryPOMDP, state::FactoryState)
    S = states(m)
    n = length(S.vtxs)
    idx = 1
    i = 0
    # for s in [state.robot_states..., state.intruder_states...]
    for s in reverse([state.robot_states..., state.intruder_states...])
        v = m.env.vtx_map[s.vtx...]
        idx += (1+S.t_max)*(v-1)*n^i
        i += 1
    end
    idx += state.t

    idx
end
function POMDPs.initialstate_distribution(m::FactoryPOMDP{E,N,M}) where {E,N,M}
    t0 = 0

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
    @assert m.n_robots == N
    possible_actions = get_possible_actions(m)
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
    # intruder_actions = map(state->rand(get_possible_actions(m,state)),s.intruder_states)
    # sp = FactoryState(
    #     robot_states = tuple(map(sa->robot_transition(m,sa...),zip(s.robot_states,a))...),
    #     intruder_states = tuple(map(sa->robot_transition(m,sa...),zip(s.intruder_states,intruder_actions))...)
    # )
    # return a distribution
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
