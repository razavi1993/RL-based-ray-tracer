# local coordinate
function local_(n::vec)
    w = unit(n)
    if abs(w.x) > 0.9
        a = vec(0, 1, 0)
    else
        a = vec(1, 0, 0)
    end
    v = unit(cross(w, a))
    u = cross(w, v)
    return u, v, w
end

# uniform sampling
function sample_direction(nl::vec, nu::Int64, nv::Int64, i::Int64, j::Int64)
    u, v, w = local_(nl)
    r = (i - rand())/nu
    s = (j - rand())/nv
    x = cos(2*π*s)*sqrt(1 - r*r)
    y = sin(2*π*s)*sqrt(1 - r*r)
    z = r
    return x*u + y*v + z*w
end

# sample lambertian reflection
function scatter(nl::vec, qvalues::Array{Float64, 2})
    patch_ind, patch_qvalue = sample_patch(qvalues)
    i, j = Tuple(patch_ind)
    nu, nv = size(qvalues)
    ω = sample_direction(nl, nu, nv, i, j)
    pdf = (patch_qvalue*nu*nv)/(2*π)
    return ω, pdf, patch_ind
end

# sarsa update
function sarsa_update(q_values::Array{Float64, 2}, max_brdf::Float64)
    s1, s2 = size(q_values)
    ξ = rand(s1, s2)
    return 2*pi*max_brdf*sum(repeat(Base.OneTo(s1), outer=(1,s2)) .- ξ)/(s1*s1*s2)
end

# radiance function
function radiance(r::ray, world::hitable, depth::Int, memory::Memory, nu::Int, nv::Int, batch_size, loss, q_network, target_network, opt::OptimizerParams, t_steps::TrainingSteps)
    put = vec(1,1,1)
    local state, action
    for i=1:depth
        hit_surface = hit(world, r, 0.0001, typemax(Float64))
        if !ismissing(hit_surface)
            if typeof(hit_surface.mater) == lambertian
                n = hit_surface.normal
                p = hit_surface.point
                next_state = reshape(hcat(transform([p.x, p.y, p.z]), [n.x, n.y, n.z]), (6,1))
                qvalues = reshape(q_network(next_state), (nu,nv))
                brdf = hit_surface.mater.color/(1.0*π)
                ω, pdf, patch_ind  = scatter(n, qvalues)
                action = patch_ind[1] + (patch_ind[2]-1)*nu
                if i > 1 && t_steps.current_step < t_steps.max_steps
                    target_qvalues = reshape(target_network(next_state), (nu,nv))
                    mse_target_qvalues = reshape(q_network(next_state), (nu,nv))
                    reward = sarsa_update(target_qvalues, maxe(brdf))
                    mse_reward = sarsa_update(mse_target_qvalues, maxe(brdf))
                    quad = Quadruple(state, action, next_state, reward, mse_reward)
                    append_remove(memory, quad)
                    if memory.min_length <= length(memory.quads)
                        inputs, indices = sample_batch(memory, batch_size, nu*nv)
                        train_network(loss, q_network, target_network, inputs, indices, memory, opt, t_steps, batch_size)
                    end
                end
                state = next_state
                cosθ = dot(ω, n)
                put = put*brdf*cosθ/pdf
                r = ray(p, ω)
            else
                if i > 1 && t_steps.current_step < t_steps.max_steps
                    reward = maxe(hit_surface.mater.emit)
                    mse_reward = maxe(hit_surface.mater.emit)
                    next_state = zeros(6,1)
                    quad = Quadruple(state, action, next_state, reward, mse_reward)
                    append_remove(memory, quad)
                    if memory.min_length <= length(memory.quads)
                        inputs, indices = sample_batch(memory, batch_size, nu*nv)
                        train_network(loss, q_network, target_network, inputs, indices, memory, opt, t_steps, batch_size)
                    end
                end
                return put*hit_surface.mater.emit
            end
        else
            if i > 1 && t_steps.current_step < t_steps.max_steps
                reward = 0.
                mse_reward = 0.
                next_state = Inf*ones(6,1)
                quad = Quadruple(state, action, next_state, reward, mse_reward)
                append_remove(memory, quad)
                if memory.min_length <= length(memory.quads)
                    inputs, indices = sample_batch(memory, batch_size, nu*nv)
                    train_network(loss, q_network, target_network, inputs, indices, memory, opt, t_steps, batch_size)
                end
            end
            return vec(0,0,0)
        end
    end
    return vec(0,0,0)
end
