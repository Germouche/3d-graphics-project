#version 330 core

uniform sampler2D diffuse_map;
uniform vec3 light_dir;
uniform vec3 k_a, k_d, k_s;
uniform vec3 w_camera_position;
uniform float s;

in vec2 frag_tex_coords;
in vec3 w_position, w_normal;

out vec4 out_color;

void main() {
    vec3 n = normalize(w_normal);
    vec3 light_obj = normalize(-light_dir);
    vec3 r = reflect(light_dir, n);
    vec3 v = normalize(w_camera_position - w_position);
    vec3 d = k_d * max(0, dot(n, light_obj));
    vec3 q = k_s * pow(dot(r, v), s);
    vec4 color = vec4(k_a + d + q, 1);
    out_color = texture(diffuse_map, frag_tex_coords) * color;
}
