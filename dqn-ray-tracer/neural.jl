using Flux
using Statistics

# q-network
q_net = fmap(f64, Chain(Dense(6, 1000, relu), Dense(1000, 108, softplus)))
target_net = fmap(f64, q_net)

# soft update
function soft_update(q_network, target_network)
    for (p1, p2) in zip(Flux.params(q_network), Flux.params(target_network))
        p2 .= 0.001*p1 .+ 0.999*p2
    end
end

# loss function
function loss(input, indices)
    x, target_values = input[1:6,:], input[7:7,:]
    return mean(abs2, q_net(x)[indices] .- target_values)
end

# training steps
mutable struct TrainingSteps
    max_steps::Int
    current_step::Int
    loss_values::Array{Float64, 1}
end

# optimizer parameters
mutable struct OptimizerParams
    γ::Float64
    decay::Float64
end

# update optimizer parameters
function update_optimizer_params(opt::OptimizerParams)
    opt.γ *= 1 - opt.decay
end

# train neural network
function train_network(loss, q_network, target_network, inputs, indices, memory::Memory, opt::OptimizerParams, t_steps::TrainingSteps, batch_size::Int)
    Flux.train!(loss, Flux.params(q_network), [(inputs, indices)], ADAM(opt.γ))
    update_optimizer_params(opt)
    soft_update(q_network, target_network)
    t_steps.current_step += 1
    t_steps.loss_values = vcat(t_steps.loss_values, loss(inputs, indices))
end
