#version 330 core
// TODO: complete the loop for TP7 exercise 1

// ---- camera geometry
uniform mat4 projection, view, model;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=4, MAX_BONES=128;
uniform mat4 bone_matrix[MAX_BONES];

// ---- vertex attributes
in vec3 position;
in vec3 normal;
in vec4 bone_ids;
in vec4 bone_weights;
in vec2 tex_coord;

// ----- interpolated attribute variables to be passed to fragment shader
out vec2 frag_tex_coords;

void main() {

    // ------ creation of the skinning deformation matrix
    mat4 skin_matrix = bone_weights.x * bone_matrix[int(bone_ids.x)]
    	 	     + bone_weights.y * bone_matrix[int(bone_ids.y)]
		     + bone_weights.z * bone_matrix[int(bone_ids.z)]
		     + bone_weights.w * bone_matrix[int(bone_ids.w)];    

    // ------ compute world and normalized eye coordinates of our vertex
    vec4 w_position4 = skin_matrix * vec4(position, 1.0);
    gl_Position = projection * view * model * w_position4;

    frag_tex_coords = tex_coord;
}
