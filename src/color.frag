#version 330 core

// output fragment color for OpenGL
in vec3 frag_color;
out vec4 out_color;
uniform vec3 global_color;

void main() {
    out_color = vec4(frag_color + global_color, 1);
}
