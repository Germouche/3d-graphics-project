#version 330 core

// input attribute variable, given per vertex
uniform mat4 projection;
uniform mat4 view;
in vec3 position;
in vec3 color;
out vec3 frag_color;

void main() {
     gl_Position = projection * view * vec4(position, 1);
     frag_color = color;
}  
