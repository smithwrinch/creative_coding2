#include "testApp.h"

testApp::~testApp()
{
	for (int i = 0; i < boids.size(); i++)
	{
		delete boids[i];
	}
}

//--------------------------------------------------------------
void testApp::setup(){


	int screenW = ofGetScreenWidth();
	int screenH = ofGetScreenHeight();

	ofBackground(0,50,50);

	// set up the boids
	for (int i = 0; i < 50; i++){
		Boid * b = new Boid();
		boids.push_back(b);
		all_boids.push_back(b);
	}

	for (int i = 0; i < 100; i++){
		Boid_Two * b = new Boid_Two(7);
		boids_two.push_back(b);
		all_boids.push_back(b);
	}

}



//--------------------------------------------------------------
void testApp::update(){

  ofVec3f min(0, 0);
	ofVec3f max(ofGetWidth(), ofGetHeight());
	// for (int i = 0; i < boids.size(); i++)
	// {
	// 	boids[i]->update(boids, min, max);
	// }
	testApp::updateBoids(boids, boids, true);
	testApp::updateBoids(boids_two, boids_two, true);
	// testApp::updateBoids(boids, boids_two, false);
	// testApp::updateBoids(boids_two, boids, false);
}

//--------------------------------------------------------------


void testApp::updateBoids(std::vector<Boid *> &otherBoids, std::vector<Boid *> &otherBoidsTwo, bool same){
	ofVec3f min(0, 0);
	ofVec3f max(ofGetWidth(), ofGetHeight());
	for (int i = 0; i < otherBoids.size(); i++)
	{
		otherBoids[i]->update(otherBoidsTwo, min, max, same);
	}
}

//--------------------------------------------------------------
void testApp::draw(){

	for (int i = 0; i < all_boids.size(); i++)
	{
		all_boids[i]->draw();
	}

}


//--------------------------------------------------------------
void testApp::keyPressed(int key){

}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}
