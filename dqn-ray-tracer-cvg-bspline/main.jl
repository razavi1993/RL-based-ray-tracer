using FileIO, ImageView, Plots, NPZ

include("vector.jl")
include("camera.jl")
include("hitable.jl")
include("surfaces.jl")
include("qtable.jl")
include("rl.jl")
include("neural.jl")
include("radiance.jl")

# render
function render(world::hitable, cam::camera, model, target_model, q_net_ps, tar_net_ps, model_ps, batch_size, loss, nu, nv, nx=300, ny=300, ns=100, depth=10)
    memory = Memory(120000, 100000, Quadruple[])
    opt = OptimizerParams(0.001, 0.005)
    t_steps = TrainingSteps(1000, 0, zeros(0))
    res = zeros(ny, nx, 3)
    image_history = zeros(ns, ny, nx, 3)
    for s in 1:ns
        for j in 1:ny
            for i in 1:nx
                r = ray(cam, (i+rand())/nx, (j+rand())/ny)
                cl = radiance(r, world, depth, memory, nu, nv, batch_size, loss, model, target_model, q_net_ps, tar_net_ps, model_ps, opt, t_steps)
                res[ny-j+1, nx-i+1, :] += [cl.x, cl.y, cl.z]
            end
        end
        image_history[s, :, :, 1:3] = sqrt.(clamp.(res./s))
    end
    return image_history[ns, :, :, 1:3], image_history, t_steps
end

# scene
function scene()
    surfaces = []
    push!(surfaces, yz_rect(-1.5, 0, 1, -1.5, 1.5, vec(1,0,0), lambertian(vec(0.75,0.25,0.25))))
    push!(surfaces, xz_rect(0, -1.5, 1.5, -1.5, 1.5, vec(0,1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, yz_rect(1.5, 0, 1, -1.5, 1.5, vec(-1,0,0), lambertian(vec(0.25,0.25,0.75))))
    push!(surfaces, xz_rect(1, -1.5, 1.5, -1.5, 1.5, vec(0,-1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(1.5, -1.5, 1.5, 0, 1, vec(0,0,-1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(-1.5, -1.5, 1.5, 0, 1, vec(0,0,1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xz_rect(0.999, -0.3, 0.3, -0.4, 0.4, vec(0,-1,0), light(vec(8,8,8))))
    push!(surfaces, sphere(0.2, vec(-0.6,0.201,0), lambertian(vec(0.75,0.35,0.35))))
    push!(surfaces, sphere(0.2, vec(0,0.201,0), lambertian(vec(0.25,0.35,0.65))))
    push!(surfaces, sphere(0.2, vec(0.6,0.201,0), lambertian(vec(0.35,0.85,0.15))))
    return hit_list(surfaces)
end

nx = 250
ny = 250
ns = 32
world = scene()
cam = camera(vec(0.0,0.5,-1.3), vec(0.0,0.5,0), vec(0,1,0), 70.0, nx/ny)
img, img_history, t_steps = render(world, cam, model, target_model, q_net_ps, target_net_ps, model_ps, 8192, loss, 10, 20, nx, ny, ns)
imshow(img)
save("C:\\Users\\javan\\Desktop\\dqn-ray-tracer-cvg-bspline\\img.png", img)
loss_values = t_steps.loss_values
loss_values = reshape(mean(reshape(loss_values, 20,50), dims=1), 50)
plot(collect(1:50), loss_values)
savefig("C:\\Users\\javan\\Desktop\\dqn-ray-tracer-cvg-bspline\\loss_values.png")

npzwrite("C:\\Users\\javan\\Desktop\\dqn-ray-tracer-cvg-bspline\\img_history.npz", img_history)
npzwrite("C:\\Users\\javan\\Desktop\\dqn-ray-tracer-cvg-bspline\\loss_values.npz", loss_values)
