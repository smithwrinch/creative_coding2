#include "boid.h"
#include <iostream>
class Boid_Two : public Boid
{
  public:
  	Boid_Two();
    Boid_Two(float mv);

	  ~Boid_Two();

    void draw();
};
