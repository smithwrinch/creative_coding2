
#include "boid_two.h"
// Derived class
Boid_Two::Boid_Two(){
}

Boid_Two::Boid_Two(float mv){
  setMaxVelocity(mv);
}

Boid_Two::~Boid_Two()
{
}

void Boid_Two::draw()
{
  // std::cout << "DRAWWWWW\n";
	ofSetColor(255, 255, 0);
	ofCircle(Boid_Two::getPosition().x, Boid_Two::getPosition().y, 3);
}
