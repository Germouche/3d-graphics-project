#version 330 core

// receiving interpolated color for fragment shader
uniform sampler2D diffuse_map;
in vec2 frag_tex_coords;

// output fragment color for OpenGL
out vec4 out_color;

void main() {
    out_color = texture(diffuse_map, frag_tex_coords);
}
