struct camera
    lower_left::vec
    horizontal::vec
    vertical::vec
    eye::vec
    u::vec
    v::vec
    w::vec
end

# create camera
function camera(look_from::vec, look_at::vec, vup::vec, vfov::Float64, aspect::Float64)
    θ = vfov * π / 180.
    height = 2.0*tan(θ/2.0)
    width = aspect*height
    w = unit(look_from - look_at)
    u = unit(cross(vup, w))
    v = cross(w, u)
    return camera(look_from - 0.5*width*u - 0.5*height*v - w, width*u, height*v, look_from, u, v, w)
end

# create ray
function ray(cam::camera, x::Float64, y::Float64)
    return ray(cam.eye, unit(cam.lower_left + x*cam.horizontal + y*cam.vertical - cam.eye))
end
