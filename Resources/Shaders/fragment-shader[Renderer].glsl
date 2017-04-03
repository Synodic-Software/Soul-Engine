 #version 430

	layout( std430, binding=0 ) buffer texB 
	{
		vec4 tex[];
	};

	layout(pixel_center_integer) in vec4 gl_FragCoord;

	in vec2 textureCoord;

    out vec4 fragment;

	uniform uvec2 screen;


    void main(){

		uvec2 pos= uvec2(textureCoord.xy*vec2(screen));

		fragment= tex[pos.x+ screen.x*pos.y];

	}