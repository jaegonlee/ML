#pragma once
#import <Foundation/Foundation.h>
#include <CoreFoundation/CoreFoundation.h>
#import <CoreML/CoreML.h>
#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "TinyYOLO.h"
#include "ofxScreenGrab.h"

#define CAM

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
#ifdef CAM
    ofVideoGrabber grabber;
#elif
    ofxScreenGrab grabber;
#endif
    ofTexture tex;
    unsigned char * pix;
};
