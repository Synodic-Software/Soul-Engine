#include "Vertex.h"

bool Vertex::operator==(const Vertex& other) const {

	return
		position == other.position &&
		normal == other.normal &&
		textureCoord == other.textureCoord &&
		velocity == other.velocity;

}