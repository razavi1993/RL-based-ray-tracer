using FileIO, ImageView, Plots, DelimitedFiles

include("vector.jl")
include("camera.jl")
include("hitable.jl")
include("surfaces.jl")
include("qtable.jl")
include("radiance.jl")

# convert to RGB image
function to_rgb(res::Array{Float64, 3})
    c, nx, ny = size(res)
    img = zeros(RGB, nx, ny)
    for i in 1:nx
        for j in 1:ny
            temp = res[:,i,j]
            img[ny-j+1,i] = RGB(temp[1], temp[2], temp[3])
        end
    end
    return img
end

# render
function render(io::IO, world::hitable, cam::camera, nx=300, ny=300, ns=100, depth=20)
    res = zeros(3, nx, ny)
    for s in 1:ns
        for j in 1:ny
            for i in 1:nx
                r = ray(cam, (i+rand())/nx, (j+rand())/ny)
                cl = radiance(r, world, depth)
                res[:, i, j] += [cl.y, cl.x, cl.z]
            end
        end
    end
    res = sqrt.(clamp.(res./ns))
    return to_rgb(res)
end

# scene
function scene()
    surfaces = []
    push!(surfaces, yz_rect(0, 0, 1, 0, 1, vec(1,0,0), lambertian(vec(0.25,0.75,0.25)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, xz_rect(0.999, 0.25, 0.75, 0.25, 0.75, vec(0,1,0), light(vec(5,5,5)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, xz_rect(0, 0, 1, 0, 1, vec(0,1,0), lambertian(vec(0.75,0.75,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, yz_rect(1, 0, 1, 0, 1, vec(-1,0,0), lambertian(vec(0.25,0.25,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, xz_rect(1, 0, 1, 0, 1, vec(0,-1,0), lambertian(vec(0.75,0.75,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, xy_rect(1, 0, 1, 0, 1, vec(0,0,-1), lambertian(vec(0.75,0.75,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, sphere(0.1, vec(0.3,0.101,0.55), lambertian(vec(0.75,0.75,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    push!(surfaces, sphere(0.1, vec(0.7,0.101,0.35), lambertian(vec(0.75,0.75,0.75)), 18, 20, create_qtable(360, 6, 8), create_paths(360, 6, 8)))
    return hit_list(surfaces)
end

nx = 500
ny = 500
ns = 256

cam = camera(vec(0.5,0.5,-1.3), vec(0.5,0.5,0), vec(0,1,0), 40.0, nx/ny)
img = render(stdout, scene(), cam, nx, ny, ns)
imshow(img)
save("E:\\Julia Projects\\rl\\scene-rl-based\\img.png", img)
