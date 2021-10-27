using StatsBase

struct Quadruple
    state::Array{Float64, 2}
    action::Int
    next_state::Array{Float64, 2}
    reward::Float64
end

# memory
struct Memory
    max_length
    min_length
    quads::Vector{Quadruple}
end

# append and remove
function append_remove(memory::Memory, quad::Quadruple)
    if length(memory.quads) < memory.max_length
        push!(memory.quads, quad)
    else
        popat!(memory.quads, 1)
        push!(memory.quads, quad)
    end
end

# sample batch
function sample_batch(memory::Memory, batch_size::Int, actions_size::Int)
    memory_length = length(memory.quads)
    sampled_indices = sample(1:memory_length, batch_size, replace=false)
    sampled_states = zeros(54, batch_size)
    sampled_rewards = zeros(1, batch_size)
    sampled_actions = zeros(Int, 1, batch_size)
    for i=1:batch_size
        sampled_states[:,i] = memory.quads[sampled_indices[i]].state
        sampled_rewards[1,i] = memory.quads[sampled_indices[i]].reward
        sampled_actions[1,i] = memory.quads[sampled_indices[i]].action + (i-1)*actions_size
    end
    return vcat(sampled_states, sampled_rewards), sampled_actions
end
