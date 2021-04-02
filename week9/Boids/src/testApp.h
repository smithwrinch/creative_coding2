#ifndef _TEST_APP
#define _TEST_APP


#include "ofMain.h"
#include <vector>
#include <iostream>
#include "boid.h"
#include "boid_two.h"

class testApp : public ofBaseApp{

public:
    ~testApp();

    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);

    std::vector<Boid *> boids;
    std::vector<Boid *> boids_two;
    std::vector<Boid *> all_boids;

private:
		void updateBoids(std::vector<Boid *> &otherBoids, std::vector<Boid *> &otherBoidsTwo, bool same);
};

#endif
